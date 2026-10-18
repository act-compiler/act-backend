use egg::*;
use std::collections::{HashSet, VecDeque};

use crate::ir::egraph::*;

/// The constant e-classes, found by a worklist
pub fn detect_constants(
  egraph: &EGraph<TensorOp, TensorInfo>,
  inputs: &HashSet<Id>,
) -> HashSet<Id> {
  let mut worklist = VecDeque::new();
  let mut constants = HashSet::new();

  // initialize worklist
  for ec in egraph.classes() {
    worklist.push_back(ec.id);
  }

  while !worklist.is_empty() {
    let eclass = worklist.pop_front().unwrap();
    if constants.contains(&eclass) {
      continue;
    }

    for en in egraph[eclass].nodes.iter() {
      if en
        .children()
        .iter()
        .all(|child_ec| constants.contains(child_ec))
        && !inputs.contains(&eclass)
      {
        constants.insert(eclass);
        for mut parent_ec in egraph[eclass].parents() {
          parent_ec = egraph.find(parent_ec);
          worklist.push_back(parent_ec);
        }
      }
    }
  }

  constants
}

/// Mark each constant class and add its DetectedConst node
pub fn annotate_constants(egraph: &mut EGraph<TensorOp, TensorInfo>, constants: &HashSet<Id>) {
  for eclass in constants.iter() {
    let mut new_metadata = egraph[*eclass].data.clone();
    new_metadata.is_const = true;
    egraph.set_analysis_data(*eclass, new_metadata);

    let mut visited: HashSet<Id> = HashSet::new();
    let repr = match op_repr(egraph, *eclass, &mut visited) {
      Some(s) => s,
      None => panic!(
        "annotate_constants: no operator-only representative found for constant eclass {}",
        eclass
      ),
    };
    let const_ec = egraph.add(TensorOp::DetectedConst(repr));
    egraph.union(*eclass, const_ec);
  }
}

// An e-class's text in Op* operators alone, or None
fn op_repr(
  egraph: &EGraph<TensorOp, TensorInfo>,
  eclass: Id,
  visited: &mut HashSet<Id>,
) -> Option<String> {
  if visited.contains(&eclass) {
    return None;
  }
  visited.insert(eclass);

  for en in egraph[eclass].nodes.iter() {
    if let Some(s) = op_repr_en(egraph, en, visited) {
      visited.remove(&eclass);
      return Some(s);
    }
  }

  visited.remove(&eclass);
  None
}

// Build Op*-only representation for a single e-node.
fn op_repr_en(
  egraph: &EGraph<TensorOp, TensorInfo>,
  en: &TensorOp,
  visited: &mut HashSet<Id>,
) -> Option<String> {
  // Operators alone (Op*): no instruction, Var or DetectedConst
  let label = match en {
    TensorOp::OpAdd(_)
    | TensorOp::OpBitcvt(_)
    | TensorOp::OpBroadcast(_, _)
    | TensorOp::OpConcat(_, _)
    | TensorOp::OpConstant(_)
    | TensorOp::OpConvert(_, _)
    | TensorOp::OpDivide(_)
    | TensorOp::OpDot(_, _)
    | TensorOp::OpExp(_)
    | TensorOp::OpEye(_)
    | TensorOp::OpOr(_)
    | TensorOp::OpReduceSum(_, _)
    | TensorOp::OpReshape(_, _)
    | TensorOp::OpShiftLeft(_)
    | TensorOp::OpShiftRightLogical(_)
    | TensorOp::OpSlice(_, _)
    | TensorOp::OpXor(_)
    | TensorOp::OpTranspose(_, _) => format!("{}", en),
    _ => return None,
  };
  let children_ids = en.children();
  if children_ids.is_empty() {
    return Some(format!("{}()", label));
  }

  let mut parts: Vec<String> = Vec::new();
  for &cid in children_ids.iter() {
    parts.push(op_repr(egraph, cid, visited)?);
  }

  Some(format!("{}({})", label, parts.join(", ")))
}

pub fn get_hbm_offset(hbm_offsets: &[(Option<Id>, i32)], eclass: Id) -> Option<i32> {
  for (buf_ec, offset) in hbm_offsets.iter() {
    if buf_ec.is_some() && buf_ec.unwrap() == eclass {
      return Some(*offset);
    }
  }
  None
}
