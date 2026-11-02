"""The Rust a precondition computes each operand's size in, as the allocator's C++ computes it"""

from .expression_writer import CheckedWriter, computes

# The helper each dividing operator of a size calls
DIVIDING = {'/': 'size_quotient', '%': 'size_remainder'}


class SizeWriter(CheckedWriter):
    """A size as Rust lets of a closure, dividing only where truncation is the oracle's floor"""

    def operation(self, operator, left, right):
        if operator in DIVIDING:
            return f'{DIVIDING[operator]}({left}, {right})?'
        return super().operation(operator, left, right)


def generate_size_checks(sizes, templates):
    """The Rust that fails a precondition where a size leaves i64, one check per size written

    A size of one attribute or one literal needs none.
    """
    checks = {}
    for tree in sizes:
        if not computes(tree):
            continue
        writer = SizeWriter()
        value = writer.visit(tree)
        checks[tree.getText()] = writer.computed('size', value) + templates.render(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.size_check.txt'
        )
    return list(checks.values())
