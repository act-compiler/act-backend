use crate::ir::egraph::TensorOp;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Buffer {
  HBM,
  // {{BUFFER_VARIANTS}}
  ANY,
}

impl std::fmt::Display for Buffer {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Buffer::HBM => write!(f, "HBM"),
      // {{BUFFER_DISPLAY_MATCH_ARMS}}
      Buffer::ANY => panic!("Buffer::ANY does not have a string representation"),
    }
  }
}

// An instruction's buffers, Some(vec![out, in1, in2, ...]), else None
pub fn buffer_assignment(en: &TensorOp) -> Option<Vec<Buffer>> {
  match en {
    // {{ISA_BUFFER_ASSIGNMENT_MATCH_ARMS}}
    TensorOp::OpSlice(_, _) => Some(vec![Buffer::ANY, Buffer::ANY]),
    TensorOp::OpConcat(_, _) => Some(vec![Buffer::ANY, Buffer::ANY, Buffer::ANY]),
    TensorOp::DetectedConst(_) => Some(vec![Buffer::HBM]),
    TensorOp::Var(_) => Some(vec![Buffer::HBM]),
    _ => None,
  }
}
