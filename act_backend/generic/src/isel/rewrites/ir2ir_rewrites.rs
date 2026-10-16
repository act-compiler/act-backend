use egg::{EGraph, Id};
use itertools::Itertools;

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};

// Each rewrite's precondition, RHS metadata and RHS shapes

pub fn precond_0(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 5);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  let i = lhs_metadata[0].shape[0] / 16;
  if (i % 2 != 0) || (i < 2) {
    return false;
  }
  if lhs_metadata[0].shape != vec![i * 16, 16] || lhs_metadata[0].dtype != Dtype::I32 {
    return false;
  }
  if lhs_metadata[1].shape != vec![16, 16] || lhs_metadata[1].dtype != Dtype::I32 {
    return false;
  }
  if lhs_metadata[2]
    != (TensorInfo {
      shape: vec![i * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    })
  {
    return false;
  }
  if lhs_metadata[3]
    != (TensorInfo {
      shape: vec![i * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    })
  {
    return false;
  }
  if lhs_metadata[4]
    != (TensorInfo {
      shape: vec![i * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    })
  {
    return false;
  }
  true
}

pub fn metadata_0(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  lhs_enodes: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 5);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let i = lhs_metadata[0].shape[0] / 16;
  // Each half's dot keeps the matched dot's dimensions
  let Some(TensorOp::OpDot(dimensions, _)) = &lhs_enodes[2] else {
    unreachable!("rewrite 0 matches a dot");
  };

  // Each half of the first dimension, as a slice
  let mut rhs_metadata = vec![None; 12];
  rhs_metadata[1] = Some(format!("{}:{}", 0, (i / 2) * 16));
  rhs_metadata[3] = Some(dimensions.clone());
  rhs_metadata[5] = Some(format!("{}:{}", 0, (i / 2) * 16));
  rhs_metadata[7] = Some(format!("{}:{}", (i / 2) * 16, i * 16));
  rhs_metadata[8] = Some(dimensions.clone());
  rhs_metadata[9] = Some(format!("{}:{}", (i / 2) * 16, i * 16));
  rhs_metadata[11] = Some("1".to_string());
  rhs_metadata
}

pub fn set_shapes_0(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 5);
  assert_eq!(rhs_eclasses.len(), 12);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let i = lhs_metadata[0].shape[0] / 16;

  egraph.set_analysis_data(
    rhs_eclasses[1],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[3],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[5],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[6],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[7],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[8],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[9],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[10],
    TensorInfo {
      shape: vec![(i / 2) * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[11],
    TensorInfo {
      shape: vec![i * 16, 16],
      dtype: Dtype::I32,
      is_const: false,
    },
  );
}

pub fn precond_1(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 3);
  true
}

pub fn metadata_1(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  lhs_enodes: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let slice_data = match &lhs_enodes[2] {
    Some(TensorOp::OpSlice(data, _)) => data.clone(),
    _ => panic!("OpSlice should have metadata!"),
  };
  let ty = lhs_metadata[1].dtype;

  let mut rhs_metadata = vec![None; 3];
  rhs_metadata[1] = Some(slice_data);
  rhs_metadata[2] = Some(ty.to_string());
  rhs_metadata
}

pub fn set_shapes_1(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 3);
  assert_eq!(rhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  egraph.set_analysis_data(
    rhs_eclasses[1],
    TensorInfo {
      shape: lhs_metadata[2].shape.clone(),
      dtype: lhs_metadata[0].dtype,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[2],
    TensorInfo {
      shape: lhs_metadata[2].shape.clone(),
      dtype: lhs_metadata[1].dtype,
      is_const: false,
    },
  );
}

pub fn precond_2(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 1);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0]
    != (TensorInfo {
      shape: vec![16, 16],
      dtype: Dtype::I8,
      is_const: false,
    })
  {
    return false;
  }
  true
}

pub fn metadata_2(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 1);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let mut metadata = lhs_metadata[0].shape.iter().join(",");
  metadata.push_str(format!(",{:?}", lhs_metadata[0].dtype).as_str());

  let mut rhs_metadata = vec![None; 3];
  rhs_metadata[1] = Some(metadata);
  // The value's dimension 1 against the identity's dimension 0
  rhs_metadata[2] = Some(";1;;0".to_string());
  rhs_metadata
}

pub fn set_shapes_2(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 1);
  assert_eq!(rhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  egraph.set_analysis_data(rhs_eclasses[1], lhs_metadata[0].clone());
  egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
}

pub fn precond_3(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 1);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0].shape.len() <= 1 || lhs_metadata[0].dtype != Dtype::U8 {
    return false;
  }
  true
}

pub fn metadata_3(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 1);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let rs2_shape = &lhs_metadata[0].shape;
  let rs1_shape = rs2_shape.iter().product::<i32>();

  let mut rhs_metadata = vec![None; 3];
  rhs_metadata[1] = Some(rs1_shape.to_string());
  rhs_metadata[2] = Some(rs2_shape.iter().join(","));
  rhs_metadata
}

pub fn set_shapes_3(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 1);
  assert_eq!(rhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let rs2_shape = &lhs_metadata[0].shape;
  let rs1_shape = rs2_shape.iter().product::<i32>();

  egraph.set_analysis_data(
    rhs_eclasses[1],
    TensorInfo {
      shape: vec![rs1_shape],
      dtype: lhs_metadata[0].dtype,
      is_const: false,
    },
  );
  egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
}

pub fn precond_4(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 1);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0].dtype == Dtype::U8 {
    return false;
  }
  true
}

pub fn metadata_4(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 1);
  let rhs_metadata = vec![None; 3];
  rhs_metadata
}

pub fn set_shapes_4(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 1);
  assert_eq!(rhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();
  let dtype_size = lhs_metadata[0].dtype.size_in_bytes();
  let bitcvt_shape = match dtype_size {
    1 => lhs_metadata[0].shape.clone(),
    _ => {
      let mut shape = lhs_metadata[0].shape.clone();
      shape.extend(vec![dtype_size]);
      shape
    }
  };

  egraph.set_analysis_data(
    rhs_eclasses[1],
    TensorInfo {
      shape: bitcvt_shape,
      dtype: Dtype::U8,
      is_const: false,
    },
  );
  egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
}

pub fn precond_5(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0].dtype != Dtype::I8 {
    return false;
  }
  if lhs_metadata[1].dtype != Dtype::I8 {
    return false;
  }
  if lhs_metadata[2].dtype != Dtype::I8 {
    return false;
  }
  true
}

pub fn metadata_5(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 3);
  let mut rhs_metadata = vec![None; 6];
  rhs_metadata[1] = Some(Dtype::I32.to_string());
  rhs_metadata[3] = Some(Dtype::I32.to_string());
  rhs_metadata[5] = Some(Dtype::I8.to_string());
  rhs_metadata
}

pub fn set_shapes_5(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 3);
  assert_eq!(rhs_eclasses.len(), 6);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  egraph.set_analysis_data(
    rhs_eclasses[1],
    TensorInfo {
      shape: lhs_metadata[0].shape.clone(),
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[3],
    TensorInfo {
      shape: lhs_metadata[0].shape.clone(),
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[4],
    TensorInfo {
      shape: lhs_metadata[0].shape.clone(),
      dtype: Dtype::I32,
      is_const: false,
    },
  );
  egraph.set_analysis_data(
    rhs_eclasses[5],
    TensorInfo {
      shape: lhs_metadata[0].shape.clone(),
      dtype: Dtype::I8,
      is_const: false,
    },
  );
}

pub fn precond_6(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0].dtype != Dtype::I32 {
    return false;
  }
  if lhs_metadata[1].dtype != Dtype::I8 {
    return false;
  }
  if lhs_metadata[2].dtype != Dtype::I32 {
    return false;
  }
  // The pair returns the value only if widened from s8
  egraph[lhs_eclasses[0]].nodes.iter().any(|node| match node {
    TensorOp::OpConvert(_, [source]) => egraph[*source].data.dtype == Dtype::I8,
    _ => false,
  })
}

pub fn metadata_6(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 3);
  let rhs_metadata = vec![None; 1];
  rhs_metadata
}

pub fn set_shapes_6(
  _: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 3);
  assert_eq!(rhs_eclasses.len(), 1);
}

pub fn precond_7(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 1);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0].dtype == Dtype::U8 {
    return false;
  }

  true
}

pub fn metadata_7(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 1);
  let rhs_metadata = vec![None; 2];
  rhs_metadata
}

pub fn set_shapes_7(
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 1);
  assert_eq!(rhs_eclasses.len(), 2);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  egraph.set_analysis_data(rhs_eclasses[1], lhs_metadata[0].clone());
}

pub fn precond_8(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  lhs_enodes: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 2);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[1].shape.len() != lhs_metadata[0].shape.len() {
    return false;
  }
  for (i, &dim) in lhs_metadata[1].shape.iter().enumerate() {
    if dim != lhs_metadata[0].shape[i] {
      return false;
    }
  }

  // The identity map keeps each element in place
  let identity: Vec<i32> = (0..lhs_metadata[0].shape.len() as i32).collect();
  TensorOp::payload_is(&lhs_enodes[1], &identity)
}

pub fn metadata_8(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 2);
  let rhs_metadata = vec![None; 1];
  rhs_metadata
}

pub fn set_shapes_8(
  _: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 2);
  assert_eq!(rhs_eclasses.len(), 1);
}

pub fn precond_9(
  egraph: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> bool {
  assert_eq!(lhs_eclasses.len(), 3);
  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses
    .iter()
    .map(|id| egraph[*id].data.clone())
    .collect();

  if lhs_metadata[0].dtype != Dtype::BF16 {
    return false;
  }

  if lhs_metadata[0].shape.len() != lhs_metadata[2].shape.len() {
    return false;
  }
  for (i, &dim) in lhs_metadata[0].shape.iter().enumerate() {
    if dim != lhs_metadata[2].shape[i] {
      return false;
    }
  }

  true
}

pub fn metadata_9(
  _: &EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  _: &[Option<TensorOp>],
) -> Vec<Option<String>> {
  assert_eq!(lhs_eclasses.len(), 3);
  let rhs_metadata = vec![None; 1];
  rhs_metadata
}

pub fn set_shapes_9(
  _: &mut EGraph<TensorOp, TensorInfo>,
  lhs_eclasses: &[Id],
  rhs_eclasses: &[Id],
) {
  assert_eq!(lhs_eclasses.len(), 3);
  assert_eq!(rhs_eclasses.len(), 1);
}
