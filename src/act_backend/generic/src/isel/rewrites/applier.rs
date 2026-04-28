use egg::*;
use itertools::Itertools;

use crate::ir::egraph::*;
use crate::isel::rewrites::{ir2ir_rewrites::*, ir2isa_rewrites::*};

use crate::PROCESSED;

pub fn get_applier(rule: &str) -> Rewrite<TensorOp, TensorInfo> {
    let fields: Vec<&str> = rule.split(':').collect();
    let name = fields[0];
    let eqn: Vec<&str> = fields[1].split("=>").collect();
    let lhs: Pattern<TensorOp> = eqn[0].parse().unwrap();
    let rhs: Pattern<TensorOp> = eqn[1].parse().unwrap();
    let precond_fn = match name {
{{ISA_PRECOND_MATCH_ARMS}}
        "0" => precond_0,
        "1" => precond_1,
        "2" => precond_2,
        "3" => precond_3,
        "4" => precond_4,
        "5" => precond_5,
        "6" => precond_6,
        "7" => precond_7,
        "8" => precond_8,
        "9" => precond_9,
        _ => panic!("No precondition function for rule {}!", name),
    };
    let metadata_fn = match name {
{{ISA_METADATA_MATCH_ARMS}}
        "0" => metadata_0,
        "1" => metadata_1,
        "2" => metadata_2,
        "3" => metadata_3,
        "4" => metadata_4,
        "5" => metadata_5,
        "6" => metadata_6,
        "7" => metadata_7,
        "8" => metadata_8,
        "9" => metadata_9,
        _ => panic!("No metadata function for rule {}!", name),
    };
    let set_shapes_fn = match name {
{{ISA_SET_SHAPES_MATCH_ARMS}}
        "0" => set_shapes_0,
        "1" => set_shapes_1,
        "2" => set_shapes_2,
        "3" => set_shapes_3,
        "4" => set_shapes_4,
        "5" => set_shapes_5,
        "6" => set_shapes_6,
        "7" => set_shapes_7,
        "8" => set_shapes_8,
        "9" => set_shapes_9,
        _ => panic!("No set_shapes function for rule {}!", name),
    };
    rewrite!(name; { lhs.clone() } => {
            ApplyRewrite { lhs, rhs, precond_fn, metadata_fn, set_shapes_fn, }
    })
}

#[derive(Debug, Clone)]
struct ApplyRewrite {
    lhs: Pattern<TensorOp>,
    rhs: Pattern<TensorOp>,
    precond_fn: fn(&EGraph<TensorOp, TensorInfo>, &Vec<Id>) -> bool,
    metadata_fn:
        fn(&EGraph<TensorOp, TensorInfo>, &Vec<Id>, &Vec<Option<TensorOp>>) -> Vec<Option<String>>,
    set_shapes_fn: fn(&mut EGraph<TensorOp, TensorInfo>, &Vec<Id>, &Vec<Id>),
}

impl Applier<TensorOp, TensorInfo> for ApplyRewrite {
    fn apply_one(
        &self,
        egraph: &mut EGraph<TensorOp, TensorInfo>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<TensorOp>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut ids = vec![];
        if PROCESSED
            .lock()
            .unwrap()
            .contains(&(rule_name, eclass, subst.clone()))
        {
            return ids;
        }
        let lhs_matches = find_matches(
            egraph,
            eclass,
            self.lhs.ast.len() - 1,
            self.lhs.ast.len(),
            &self.lhs.ast,
        );
        PROCESSED
            .lock()
            .unwrap()
            .insert((rule_name, eclass, subst.clone()));

        for lhs_match in lhs_matches {
            let (lhs_eclasses, lhs_enodes): (Vec<_>, Vec<_>) = lhs_match.into_iter().unzip();
            let precond = (self.precond_fn)(egraph, &lhs_eclasses);
            if !precond {
                continue;
            }

            let rhs_metadata = (self.metadata_fn)(egraph, &lhs_eclasses, &lhs_enodes);
            let rhs = self.rhs.ast.as_ref();
            let mut rhs_eclasses = vec![0.into(); rhs.len()];
            assert_eq!(rhs_eclasses.len(), rhs_metadata.len());
            let id = apply_pat(egraph, rhs, &mut rhs_eclasses, &rhs_metadata, subst);
            (self.set_shapes_fn)(egraph, &lhs_eclasses, &rhs_eclasses);

            if egraph.union(eclass, id) {
                ids.push(id);
            }
        }
        ids
    }

    fn vars(&self) -> Vec<Var> {
        self.rhs.vars()
    }
}

/// Creates the enodes for the RHS of a rewrite.
/// Returns the Id of the RHS root.
fn apply_pat(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    rhs_pat: &[ENodeOrVar<TensorOp>],
    rhs_eclasses: &mut [Id],
    rhs_metadata: &Vec<Option<String>>,
    subst: &Subst,
) -> Id {
    for (i, pat_node) in rhs_pat.iter().enumerate() {
        let id = match pat_node {
            ENodeOrVar::Var(w) => subst[*w],
            ENodeOrVar::ENode(e) => {
                let mut en = e
                    .clone()
                    .map_children(|child| rhs_eclasses[usize::from(child)]);
                let metadata = rhs_metadata[i].clone();
                en.set_metadata(metadata);
                egraph.add(en.clone())
            }
        };
        rhs_eclasses[i] = id;
    }

    *rhs_eclasses.last().unwrap()
}

/// Return all sequences of eclasses and enodes corresponding to a pattern match at eclass.
/// Each sequence will be in post-order.
fn find_matches(
    egraph: &EGraph<TensorOp, TensorInfo>,
    eclass: Id,
    pat_idx: usize,
    limit: usize,
    pat: &[ENodeOrVar<TensorOp>],
) -> Vec<Vec<(Id, Option<TensorOp>)>> {
    let mut matches = vec![];
    let pat_node = pat.iter().nth(pat_idx).unwrap();
    match pat_node {
        ENodeOrVar::Var(_) => {
            if limit > 0 {
                matches.push(vec![(eclass, None)]);
            }
        }
        ENodeOrVar::ENode(e) => {
            for en in egraph[eclass].nodes.iter() {
                // only consider enodes in root eclass of the same variant as e
                if e.discriminant() != en.discriminant() {
                    continue;
                }
                let mut child_matches = vec![];
                for (i, child_ec) in en.children().iter().enumerate() {
                    child_matches.push(find_matches(
                        egraph,
                        *child_ec,
                        e.children()[i].into(),
                        limit - 1,
                        pat,
                    ));
                }
                if child_matches.iter().all(|x| !x.is_empty()) {
                    for combination in child_matches.into_iter().multi_cartesian_product() {
                        let mut seq: Vec<_> = combination.into_iter().flatten().collect();
                        if seq.len() < limit {
                            seq.push((eclass, Some(en.clone())));
                            matches.push(seq);
                        }
                    }
                }
            }
        }
    }
    matches
}
