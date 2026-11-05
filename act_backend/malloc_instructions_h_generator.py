"""Generator for instructions.h - templates the generic file with instruction classes"""

import os
import re

from antlr4.tree.Tree import TerminalNode
from antlr4.tree.Trees import Trees
from taidl.antlr4 import IDLV2ExprParser, IDLV2ExprVisitor

from .expression_writer import chain, computes, source
from .instruction_info import get_attribute_name, get_type_names
from .template_loader import call_lines, fill_placeholders, fits, get_backend_template_loader


def generate_size(size, name):
    """A size's local as clang-format lays it out: on one line, after its =, or after operators"""
    value = source(size, cpp=True)
    for local in (f'    int64 {name} = {value};', f'    int64 {name} =\n        {value};'):
        if fits(local):
            return local
    # Each operand on the line it fits, an operator ending a line
    operands, operators = chain(size)
    afters = [*(f' {operator}' for operator in operators), ';']
    first = f'    int64 {name} = {operands[0]}{afters[0]}'
    if fits(first):
        # The rest aligned after the =
        lines, indent = [first], ' ' * len(f'    int64 {name} = ')
    else:
        # All at the continuation indent below it
        lines, indent = [f'    int64 {name} =', f'        {operands[0]}{afters[0]}'], ' ' * 8
    for operand, after in zip(operands[1:], afters[1:]):
        if fits(f'{lines[-1]} {operand}{after}'):
            lines[-1] += f' {operand}{after}'
        else:
            lines.append(f'{indent}{operand}{after}')
    return '\n'.join(lines)


def generate_ranges(ranges):
    """The ranges' return as clang-format lays it out: on one line, else a range to a line

    The first range after the return's brace, or every one below it where their breaks are
    fewer, each range broken only as far as it must.
    """
    # Every range on the return's line where they fit
    pairs = ', '.join(f'{{{address}, MAKE_SUM({address}, {size})}}' for address, size in ranges)
    if fits(f'    return {{{pairs}}};'):
        return f'    return {{{pairs}}};'
    layouts = []
    for below in (False, True):
        # Each range's brace after the return's, else at the continuation indent
        column = 8 if below else 12
        lines = ['    return {'] if below else []
        for k, (address, size) in enumerate(ranges):
            lead = '    return {' if k == 0 and not below else ' ' * column
            tail = '};' if k == len(ranges) - 1 else ','
            # On one line, broken after its start, then inside its sum
            opening, pad = f'{lead}{{{address},', ' ' * (column + 1)
            forms = [
                [f'{lead}{{{address}, MAKE_SUM({address}, {size})}}{tail}'],
                [opening, f'{pad}MAKE_SUM({address}, {size})}}{tail}'],
                [opening, f'{pad}MAKE_SUM({address},', f'{pad}         {size})}}{tail}'],
            ]
            lines += next((form for form in forms if fits('\n'.join(form))), forms[-1])
        layouts.append(lines)
    # Below where it takes no more lines, its ranges breaking less
    after, below = layouts
    return '\n'.join(below if len(below) <= len(after) else after)


def names_address(tree):
    """Whether a parsed value or constraint names an address attribute"""
    return bool(Trees.findAllTokenNodes(tree, IDLV2ExprParser.ADDRESSING))


def divisors(tree):
    """Each operand a parsed value or constraint divides or takes a remainder by"""
    return [
        operand
        for run in Trees.descendants(tree)
        if isinstance(run, IDLV2ExprParser.MultiplicativeContext)
        for position, operand in enumerate(run.unary()[1:], 1)
        if run.getChild(2 * position - 1).getText() in ('/', '%')
    ]


def divides(tree):
    """Whether a parsed value divides or takes a remainder"""
    tokens = (IDLV2ExprParser.SLASH, IDLV2ExprParser.PERCENT)
    return any(Trees.findAllTokenNodes(tree, token) for token in tokens)


def literal(tree):
    """An operand's integer literal, brackets aside; else None"""
    tokens = [
        node
        for node in Trees.descendants(tree)
        if isinstance(node, TerminalNode) and node.getText() not in ('(', ')')
    ]
    if len(tokens) == 1 and tokens[0].getSymbol().type == IDLV2ExprParser.INT:
        return tokens[0].getText()
    return None


def solver_steps(run):
    """A run's first operand, then each operator with its operand, a leading literal factor last"""
    parts = list(run.getChildren())
    first = parts[0]
    steps = [(operator.getText(), operand) for operator, operand in zip(parts[1::2], parts[2::2])]
    # The solver's product takes a literal second
    if steps and steps[0][0] == '*' and literal(first) and not literal(steps[0][1]):
        first, steps[0] = steps[0][1], ('*', first)
    return first, steps


# The solver's call for each arithmetic operator, a quotient and a remainder rounded down
SOLVER_OPERATORS = {
    '+': 'MAKE_SUM',
    '-': 'solver->MakeDifference',
    '*': 'MAKE_PROD',
    '/': 'floor_div',
    '%': 'floor_mod',
}


class SolverWriter(IDLV2ExprVisitor):
    """A value as the solver's C++, each operand naming no address a known integer

    An operand that divides is the solver's to compute, rounded down as the language does.
    """

    def __init__(self):
        super().__init__()
        # Each known integer's tree
        self.known = []

    def visit(self, tree):
        # An operand naming no address and dividing nothing is a known integer
        if names_address(tree) or divides(tree):
            return super().visit(tree)
        self.known.append(tree)
        return f'MAKE_INT_KNOWN({source(tree, cpp=True)})'

    def visitExpression(self, ctx):
        return self.visit(ctx.additive())

    def operation(self, operator, value, operand):
        """One operator's call on the value so far, a literal factor bare"""
        factor = literal(operand) if operator == '*' else None
        # Never 0, which C++ also reads as a null pointer
        if factor == '0':
            factor = None
        return f'{SOLVER_OPERATORS[operator]}({value}, {factor or self.visit(operand)})'

    def chain(self, ctx):
        # Each operator a call on the value so far
        first, steps = solver_steps(ctx)
        value = self.visit(first)
        for operator, operand in steps:
            value = self.operation(operator, value, operand)
        return value

    def visitAdditive(self, ctx):
        return self.chain(ctx)

    def visitMultiplicative(self, ctx):
        return self.chain(ctx)

    def visitUnary(self, ctx):
        if ctx.MINUS() is None:
            return self.visit(ctx.atom())
        return f'solver->MakeOpposite({self.visit(ctx.unary())})'

    def visitAtom(self, ctx):
        # A bracket nests as a call does, an address attribute by its generated name
        if ctx.expression() is not None:
            return self.visit(ctx.expression())
        return get_attribute_name(ctx.reference().getText()[3:])


def assigned(target, value):
    """An assignment as clang-format lays it out, on one line or after its =; else None"""
    for line in (f'    {target} = {value};', f'    {target} =\n        {value};'):
        if fits(line):
            return line + '\n'
    return None


def solver_locals(name, tree, target=None):
    """The C++ that leaves a value in a local, one too wide for a line built an operand at a time"""
    target = target or f'IntExpr *{name}'
    value = SolverWriter().visit(tree)
    line = assigned(target, value)
    if line is not None:
        return line
    # A bracket or a lone operand stands for what it holds, a negation negated after
    if isinstance(tree, IDLV2ExprParser.ExpressionContext):
        return solver_locals(name, tree.additive(), target)
    if isinstance(tree, IDLV2ExprParser.AtomContext):
        if tree.expression() is None:
            if '(' not in value:
                # A bare member after its =, however wide
                return f'    {target} =\n        {value};\n'
            # One wide attribute, below its call's parenthesis
            call, argument = value.split('(', 1)
            return f'    {target} = {call}(\n        {argument};\n'
        return solver_locals(name, tree.expression(), target)
    if isinstance(tree, IDLV2ExprParser.UnaryContext):
        if tree.MINUS() is None:
            return solver_locals(name, tree.atom(), target)
        negation = assigned(name, f'solver->MakeOpposite({name})')
        return solver_locals(name, tree.unary(), target) + negation
    # A sum or a product: its first operand, then each operator's call
    first, steps = solver_steps(tree)
    lines = solver_locals(name, first, target)
    for position, (operator, operand) in enumerate(steps, 1):
        line = assigned(name, SolverWriter().operation(operator, name, operand))
        if line is None:
            # A wide operand in a local of its own
            lines += solver_locals(f'{name}_{position}', operand)
            line = assigned(name, f'{SOLVER_OPERATORS[operator]}({name}, {name}_{position})')
        lines += line
    return lines


def known_integers(tree):
    """The known integers of a start, or of a comparison the solver takes, that hold an operator"""
    writer = SolverWriter()
    for value in tree.additive() if isinstance(tree, IDLV2ExprParser.ComparisonContext) else [tree]:
        writer.visit(value)
    return [part for part in writer.known if computes(part)]


def generate_start(address, name):
    """A scaled start's local as clang-format lays it out: on one line, after its =, or after ("""
    for local in (
        f'    IntExpr *{name} = {address};',
        f'    IntExpr *{name} =\n        {address};',
    ):
        if fits(local):
            return local
    # The product's arguments below its parenthesis
    arguments = address.removeprefix('MAKE_PROD(')
    return f'    IntExpr *{name} = MAKE_PROD(\n        {arguments};'


def generate_get_h_method(index, starts, sizes, templates):
    addresses = [SolverWriter().visit(start) for start in starts]
    # A range per addressed dimension: the start, then start plus size
    ranges = '}, {'.join(
        f'{address}, MAKE_SUM({address}, {source(size, cpp=True)})'
        for address, size in zip(addresses, sizes)
    )
    method = templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_h_method.txt', index=index, ranges=ranges
    )
    # A start the solver computes, beyond an attribute or one product, is a local of its own
    computed = [
        names_address(start) and not re.fullmatch(r'\w+|MAKE_PROD\(\w+, \d+\)', address)
        for start, address in zip(starts, addresses)
    ]
    if fits(method) and not any(computed):
        return method
    # Each such start, each scaled one and each size first, as a local of its own
    suffixes = [''] if len(sizes) == 1 else [str(k) for k in range(len(sizes))]
    firsts, lines = [], []
    for start, address, lifted, suffix in zip(starts, addresses, computed, suffixes):
        if lifted:
            # Built a step at a time where it is too wide for its line
            lines.append(solver_locals(f'start{suffix}', start).rstrip('\n'))
            address = f'start{suffix}'
        elif address.startswith('MAKE_PROD('):
            lines.append(generate_start(address, f'start{suffix}'))
            address = f'start{suffix}'
        firsts.append(address)
    names = [f'size{suffix}' for suffix in suffixes]
    lines += [generate_size(size, name) for size, name in zip(sizes, names)]
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_h_method_wrapped.txt',
        index=index,
        locals='\n'.join(lines),
        ranges=generate_ranges(list(zip(firsts, names))),
    )


# Each comparison's solver constraint, and its flag inside an or
COMPARISONS = {
    '==': ('MakeEquality', 'MakeIsEqualVar'),
    '!=': ('MakeNonEquality', 'MakeIsDifferentVar'),
    '<': ('MakeLess', 'MakeIsLessVar'),
    '<=': ('MakeLessOrEqual', 'MakeIsLessOrEqualVar'),
    '>': ('MakeGreater', 'MakeIsGreaterVar'),
    '>=': ('MakeGreaterOrEqual', 'MakeIsGreaterOrEqualVar'),
}


class ConditionWriter:
    """The C++ that pushes conditions to the solver, as clang-format lays it out

    A comparison is the solver's own, an and each of its comparisons; under an or each part is a
    flag, an or's at least one set and an and's all set.
    """

    def __init__(self):
        self.lines = ''
        # Comparisons so far, naming their wide sides
        self.position = 0
        # The constraint's vectors of flags, each named from its base
        self.base, self.vectors = '', []

    def push(self, constraint, base):
        """Push the constraints that hold one constraint, its vectors of flags named from base"""
        self.base, self.vectors = base, []
        self.hold(constraint)

    def vector(self):
        """Declare a vector of flags, named by the constraint and its count; return its name"""
        name = self.base if not self.vectors else f'{self.base}_{len(self.vectors)}'
        self.vectors.append(name)
        self.lines += f'    std::vector<IntVar *> {name};\n'
        return name

    def hold(self, condition):
        """Push the constraints that hold a condition"""
        conjunctions = condition.conjunction()
        if len(conjunctions) == 1:
            for comparison in conjunctions[0].comparison():
                if comparison.condition() is not None:
                    self.hold(comparison.condition())
                else:
                    self.compare(comparison, 'constraints')
            return
        # At least one part holds: their flags' sum
        flags = self.vector()
        for conjunction in conjunctions:
            self.flag(conjunction.comparison(), flags)
        least = f'solver->MakeGreaterOrEqual(solver->MakeSum({flags}), int64_t(1))'
        self.lines += f'    constraints.push_back({least});\n'

    def flag(self, comparisons, flags):
        """Push onto flags whether each of an and's comparisons holds"""
        if len(comparisons) > 1:
            parts = self.vector()
            for comparison in comparisons:
                self.flag([comparison], parts)
            count = len(comparisons)
            every = f'solver->MakeIsEqualCstVar(solver->MakeSum({parts}), int64_t({count}))'
            self.push_flag(flags, every, parts)
            return
        (comparison,) = comparisons
        if comparison.condition() is None:
            self.compare(comparison, flags)
            return
        # A bracketed condition: its one and, else at least one of its parts
        conjunctions = comparison.condition().conjunction()
        if len(conjunctions) == 1:
            self.flag(conjunctions[0].comparison(), flags)
            return
        parts = self.vector()
        for conjunction in conjunctions:
            self.flag(conjunction.comparison(), parts)
        least = f'solver->MakeIsGreaterOrEqualCstVar(solver->MakeSum({parts}), int64_t(1))'
        self.push_flag(flags, least, parts)

    def push_flag(self, flags, value, parts):
        """Push a flag of parts onto flags, from a local of its own where the line is too wide"""
        line = f'    {flags}.push_back({value});'
        if not fits(line):
            self.lines += assigned(f'IntVar *{parts}_held', value)
            line = f'    {flags}.push_back({parts}_held);'
        self.lines += line + '\n'

    def compare(self, comparison, lead):
        """Push a comparison onto lead: the solver's constraint, or its flag in a vector"""
        plain, flag = COMPARISONS[comparison.compare().getText()]
        call = f'{lead}.push_back(solver->{plain if lead == "constraints" else flag}('
        trees = comparison.additive()
        sides = [SolverWriter().visit(side) for side in trees]
        # A line too wide computes its wider side first, then the other
        for k in sorted((0, 1), key=lambda k: -len(sides[k])):
            if fits(f'    {call}{", ".join(sides)}));'):
                break
            name = f'{("left", "right")[k]}{self.position}'
            self.lines += solver_locals(name, trees[k])
            sides[k] = name
        self.lines += f'    {call}{", ".join(sides)}));\n'
        self.position += 1


def generate_constraints(constraints):
    """The C++ that pushes each constraint to the solver, as clang-format lays it out"""
    writer = ConditionWriter()
    for index, tree in enumerate(constraints):
        writer.push(tree, f'holds{index}')
    return writer.lines


def generate_constructor_packed(class_name, op_name, attributes):
    """A constructor too wide for its two templates, packed as clang-format packs it"""
    parameters = ['Tensor *y', 'std::vector<Tensor *> X', *(f'int64 {a}' for a in attributes)]
    declarator = call_lines(f'  {class_name}(', parameters, ')', 2, declaration=True)
    # Initializers as many to a line as fit, one too wide broken after its parenthesis
    pieces = [('Instruction(', f'"{op_name}", y, X)'), *((f'{a}(', f'{a})') for a in attributes)]
    lines = ['      :']
    for k, (head, value) in enumerate(pieces):
        value += ' {' if k == len(pieces) - 1 else ','
        column = len(lines[-1]) + 5
        if fits(f'{lines[-1]} {head}{value}'):
            lines[-1] += f' {head}{value}'
        elif fits(f'        {head}{value}'):
            lines.append(f'        {head}{value}')
        elif fits(f'{lines[-1]} {head}') and fits(' ' * column + value):
            lines[-1] += f' {head}'
            lines.append(' ' * column + value)
        else:
            lines += [f'        {head}', f'            {value}']
    return '\n'.join([declarator, *lines])


def generate_get_region_method(index, sizes, units, templates):
    """An operand's region: its size along each addressed dimension, then each unit dimension"""
    extents = [*(source(size, cpp=True) for size in sizes), *map(str, units)]
    method = templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_region_method.txt',
        index=index,
        extents=', '.join(extents),
    )
    if fits(method):
        return method
    # Each size a local of its own, named as a range's is
    names = ['size'] if len(sizes) == 1 else [f'size{k}' for k in range(len(sizes))]
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_region_method_wrapped.txt',
        index=index,
        locals='\n'.join(generate_size(size, name) for size, name in zip(sizes, names)),
        extents=', '.join([*names, *map(str, units)]),
    )


def generate_instruction_class(instruction, class_name, templates):
    """Generate C++ instruction class from instruction definition"""
    op_name = instruction.instruction
    comp_attrs = instruction.comp_attr
    params = instruction.parameters
    # Each attribute's generated name; the assembly prints the spec's
    names = {attr: get_attribute_name(attr) for attr in [*comp_attrs, *params]}

    # Generate members
    comp_attr_members = ''
    for attr in comp_attrs:
        comp_attr_members += f'  int64 {names[attr]};\n'

    # A blank line parts the two kinds of member
    addr_var_members = '\n' if comp_attrs else ''
    for param in params:
        member = f'  IntVar *{names[param]} = MAKE_INT_VAR(0, INT64_MAX);'
        if not fits(member):
            # Broken after its =
            member = f'  IntVar *{names[param]} =\n      MAKE_INT_VAR(0, INT64_MAX);'
        addr_var_members += member + '\n'

    # get_h methods: the output's first (get_h0), then the inputs'
    get_h_methods = ''
    get_h_cases = ''

    # The output's ranges first, then each input's
    operands = [*instruction.instr_output_asts, *instruction.instr_input_asts]
    for h_index, (_, starts, sizes) in enumerate(operands):
        get_h_methods += generate_get_h_method(h_index, starts, sizes, templates)
        get_h_cases += templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.get_h_case.txt', index=h_index
        )

    # Generate constructor parameters
    comp_attr_params = ''
    if comp_attrs:
        comp_attr_params = ', ' + ', '.join(f'int64 {names[attr]}' for attr in comp_attrs)

    # Generate constructor initialization
    comp_attr_init = ''
    if comp_attrs:
        comp_attr_init = ', ' + ', '.join(f'{names[attr]}({names[attr]})' for attr in comp_attrs)

    # The constructor on one line, else broken before its initializers
    constructor = templates.render_fitting(
        'instructions.h.INSTRUCTION_CLASSES.constructor.txt',
        class_name=class_name,
        op_name=op_name,
        parameters=comp_attr_params,
        initializers=comp_attr_init,
    )
    if not fits(constructor):
        constructor = generate_constructor_packed(
            class_name, op_name, [names[attr] for attr in comp_attrs]
        )

    # The text names each attribute's value, a line each
    values = [(attr, names[attr]) for attr in comp_attrs]
    values += [(param, f'{names[param]}->Value()') for param in params]
    str_pieces = ''
    for index, (attr, value) in enumerate(values):
        str_pieces += templates.render_fitting(
            'instructions.h.INSTRUCTION_CLASSES.str_piece.txt',
            label=(', ' if index else '') + f'{attr} = ',
            value=value,
        )
        str_pieces += '\n'

    # The constraints naming an address, the selector checking the rest
    constraints = [tree for tree in instruction.constraint_asts if names_address(tree)]

    # Address variables on one line, else pushed a line each
    variables = [names[param] for param in params]
    get_int_var = templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_int_var.txt', variables=', '.join(variables)
    )
    if not fits(get_int_var):
        pushes = ''.join(
            templates.render_lines(
                'instructions.h.INSTRUCTION_CLASSES.get_int_var_push.txt', variable=variable
            )
            for variable in variables
        )
        get_int_var = templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.get_int_var_pushed.txt', pushes=pushes
        )

    # Each operand's region, numbered as its ranges are
    get_region_methods = ''
    get_region_cases = ''
    for index, (buffer, _, sizes) in enumerate(operands):
        units = instruction.find_data_model(buffer).unit_dim
        get_region_methods += generate_get_region_method(index, sizes, units, templates)
        get_region_cases += templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.get_region_case.txt', index=index
        )

    # Generate complete class using template
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.class.txt',
        CLASS_NAME=class_name,
        COMP_ATTR_MEMBERS=comp_attr_members,
        ADDR_VAR_MEMBERS=addr_var_members,
        GET_H_METHODS=get_h_methods,
        GET_REGION_METHODS=get_region_methods,
        CONSTRUCTOR=constructor,
        STR_PIECES=str_pieces,
        GET_H_CASES=get_h_cases,
        CONSTRAINTS=generate_constraints(constraints),
        GET_INT_VAR=get_int_var,
        GET_REGION_CASES=get_region_cases,
    )


def generate_instructions_file(backend_gen_dir, instructions):
    """Template instructions.h from generic file"""
    templates = get_backend_template_loader()

    # Read generic template file
    generic_file = os.path.join(backend_gen_dir, 'cpp', 'malloc', 'include', 'instructions.h')
    with open(generic_file, 'r') as f:
        content = f.read()

    # Generate all instruction classes, a blank line between two
    class_names = get_type_names([instruction.instruction for instruction in instructions])
    classes = '\n'.join(
        generate_instruction_class(instruction, class_names[instruction.instruction], templates)
        for instruction in instructions
    ).rstrip('\n')

    # Replace placeholder
    content = fill_placeholders(content, {'INSTRUCTION_CLASSES': classes}, generic_file)

    # Write back
    with open(generic_file, 'w') as f:
        f.write(content)
