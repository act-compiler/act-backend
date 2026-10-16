// The e-graph nodes of each kernel line, by its operation

use std::collections::HashSet;

use egg::{EGraph, Id};

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};

use crate::isel::initializer::lib::EggSymbolTable;
use crate::isel::initializer::shapes::{
  apply_transpose_order, byte_view, get_bytes, get_result_shape, get_tiled_shape,
  get_transpose_order_param, get_transpose_order_result, shape_text, shape_to_vec,
};

macro_rules! binary_op_match {
    ($op_name:expr, $lhs:expr, $rhs:expr, $table:expr, $egraph:expr, {
        $($key:literal => $variant:path),* $(,)?
    }) => {
        match $op_name {
            $(
                $key => {
                    $egraph.add($variant([
                        lookup($table, $lhs)?,
                        lookup($table, $rhs)?,
                    ]))
                }
            )*
            // The parser builds a binary operation for these words alone
            other => unreachable!("binary operation {}", other),
        }
    };
}

// A kernel name's e-class, or why none is defined
fn lookup(table: &EggSymbolTable, symbol: &str) -> Result<Id, String> {
  table.get(symbol).copied().ok_or_else(|| {
    format!(
      "the kernel reads {}, which no earlier line of it defines",
      symbol
    )
  })
}

// Give a node its type, shape and kernel name
pub fn bind_node(
  tsymbol: &str,
  node: Id,
  shape: Vec<i32>,
  dtype: Dtype,
  is_const: bool,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) {
  egraph.set_analysis_data(
    node,
    TensorInfo {
      shape,
      dtype,
      is_const,
    },
  );
  table.insert(tsymbol.to_string(), node);
}

pub fn add_parameter(
  tsymbol: &str,
  tshape: &str,
  ttile: &str,
  ttype: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  inputs: &mut HashSet<Id>,
  table: &mut EggSymbolTable,
) -> Result<Id, String> {
  let dtype = Dtype::try_from(ttype)?;
  let dtype_size = dtype.size_in_bytes();

  let bytes = get_bytes(tshape, dtype_size)?;
  if ttile.is_empty() {
    // no tile supplied case
    let var_node = egraph.add(TensorOp::Var(tsymbol.to_string()));
    egraph.set_analysis_data(
      var_node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let shape_with_size = byte_view(tshape, dtype_size);

    let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [var_node]));
    let reshape_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(tshape)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    table.insert(tsymbol.to_string(), node);
    inputs.insert(var_node);
    Ok(var_node)
  } else {
    let var_node = egraph.add(TensorOp::Var(tsymbol.to_string()));
    egraph.set_analysis_data(
      var_node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let tiled_shape = get_tiled_shape(tshape, ttile)?;

    let mut shape_with_size = tiled_shape.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [var_node]));
    let reshape_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(&tiled_shape)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    let transpose_order = get_transpose_order_param(tshape.split(',').count() as i32);
    let node = egraph.add(TensorOp::OpTranspose(transpose_order.clone(), [node]));
    // Each tile dimension beside its outer one
    let transposed_str = apply_transpose_order(&tiled_shape, &transpose_order);
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: shape_to_vec(&transposed_str)?,
        dtype,
        is_const: false,
      },
    );

    // The tiles joined back into the shape
    let node = egraph.add(TensorOp::OpReshape(tshape.to_string(), [node]));
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: shape_to_vec(tshape)?,
        dtype,
        is_const: false,
      },
    );

    table.insert(tsymbol.to_string(), node);
    inputs.insert(var_node);
    Ok(var_node)
  }
}

pub fn add_constant(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  value: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpConstant(value.to_string()));
  bind_node(tsymbol, node, shape, dtype, true, egraph, table);
  Ok(())
}

pub fn add_broadcast(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  dimensions: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  // The result shape, then the dimension map
  let node = egraph.add(TensorOp::OpBroadcast(
    format!("{};{}", tshape, dimensions),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

// One argument per field of the kernel's line
#[allow(clippy::too_many_arguments)]
pub fn add_binary_op(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  op_name: &str,
  lhs: &str,
  rhs: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = binary_op_match!(op_name, lhs, rhs, table, egraph, {
      "add" => TensorOp::OpAdd,
      "xor" => TensorOp::OpXor,
      "shift-right-logical" => TensorOp::OpShiftRightLogical,
      "or" => TensorOp::OpOr,
      "shift-left" => TensorOp::OpShiftLeft,
      "dot" => TensorOp::OpDot,
      "divide" => TensorOp::OpDivide
  });
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_slice(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  slices: &[String],
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;
  let source = lookup(table, source)?;
  // The dimensions the bounds lie in, as the allocator checks
  let dims = egraph[source].data.shape.clone();
  if slices.len() != dims.len() {
    return Err(format!(
      "{} slices a value of rank {} with bounds for rank {}",
      tsymbol,
      dims.len(),
      slices.len()
    ));
  }

  // The reason for a dimension's entry that gives no bounds
  let malformed = |slice: &str| {
    format!(
      "{} slices with '{}', which is not a [start:limit:stride] range",
      tsymbol, slice
    )
  };

  // Each dimension's start:limit, its stride one
  let mut bounds = vec![];
  for (slice, dim) in slices.iter().zip(dims) {
    let parts: Vec<&str> = slice
      .trim_matches(|c| c == '[' || c == ']')
      .split(':')
      .collect();
    let (start, limit) = match parts[..] {
      [start, limit] | [start, limit, "1"] => (start, limit),
      [_, _, stride] => {
        return Err(format!(
          "{} slices with the stride {}, where the backend supports only a stride of one",
          tsymbol, stride
        ))
      }
      _ => return Err(malformed(slice)),
    };
    let (Ok(start), Ok(limit)) = (start.parse::<i32>(), limit.parse::<i32>()) else {
      return Err(malformed(slice));
    };
    if start < 0 || start >= limit || limit > dim {
      return Err(format!(
        "{} slices {} of a dimension of {}, where 0 <= start < limit <= {}",
        tsymbol, slice, dim, dim
      ));
    }
    bounds.push((start, limit));
  }
  // The result's shape is each range's length
  let sizes: Vec<i32> = bounds.iter().map(|(start, limit)| limit - start).collect();
  if sizes != shape {
    return Err(format!(
      "{} gives the shape [{}] to a slice of the shape [{}]",
      tsymbol,
      shape_text(&shape),
      shape_text(&sizes)
    ));
  }
  let slice_str = bounds
    .iter()
    .map(|(start, limit)| format!("{}:{}", start, limit))
    .collect::<Vec<_>>()
    .join(",");

  let node = egraph.add(TensorOp::OpSlice(slice_str, [source]));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_concatenate(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  inputs: &[String],
  dimensions: &[u32],
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  if inputs.len() != 2 {
    let values = if inputs.len() == 1 { "value" } else { "values" };
    return Err(format!(
      "{} concatenates {} {}, where a concatenate takes 2",
      tsymbol,
      inputs.len(),
      values
    ));
  }
  let (lhs, rhs) = (lookup(table, &inputs[0])?, lookup(table, &inputs[1])?);
  // The shapes join along the dimension, as the allocator checks
  let (lhs_shape, rhs_shape) = (&egraph[lhs].data.shape, &egraph[rhs].data.shape);
  let axis = dimensions[0] as usize;
  if axis >= lhs_shape.len() {
    return Err(format!(
      "{} concatenates values of rank {} along dimension {}",
      tsymbol,
      lhs_shape.len(),
      axis
    ));
  }
  let mut joined = lhs_shape.clone();
  joined[axis] += rhs_shape.get(axis).copied().unwrap_or(0);
  let alike = lhs_shape.len() == rhs_shape.len()
    && (0..lhs_shape.len()).all(|d| d == axis || lhs_shape[d] == rhs_shape[d]);
  if !alike || joined != shape {
    return Err(format!(
      "{} concatenates [{}] and [{}] along dimension {}, which does not give its shape [{}]",
      tsymbol,
      shape_text(lhs_shape),
      shape_text(rhs_shape),
      axis,
      shape_text(&shape)
    ));
  }
  let dim_str = dimensions
    .iter()
    .map(|d| d.to_string())
    .collect::<Vec<String>>()
    .join(",");
  let node = egraph.add(TensorOp::OpConcat(dim_str.clone(), [lhs, rhs]));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_reshape(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpReshape(
    tshape.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_transpose(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  dimensions: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpTranspose(
    dimensions.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_convert(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpConvert(
    ttype.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_reduce(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  dimensions: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpReduceSum(
    dimensions.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_exponential(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpExp([lookup(table, source)?]));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub fn add_root_operations(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  ttile: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &EggSymbolTable,
  root: &mut Option<Id>,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let dtype_size = dtype.size_in_bytes();
  let bytes = get_bytes(tshape, dtype_size)?;

  if ttile.is_empty() {
    let shape_with_size = byte_view(tshape, dtype_size);

    let node = lookup(table, tsymbol)?;

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpReshape(bytes.to_string(), [node]));
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    *root = Some(node);
  } else {
    let node = lookup(table, tsymbol)?;
    let tiled_shape = get_result_shape(tshape, ttile)?;

    let node = egraph.add(TensorOp::OpReshape(tiled_shape.clone(), [node]));
    let reshape_vec = shape_to_vec(&tiled_shape)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    let transpose_order = get_transpose_order_result(tshape.split(',').count() as i32);
    let node = egraph.add(TensorOp::OpTranspose(transpose_order.clone(), [node]));

    let transposed_reshape_str = apply_transpose_order(&tiled_shape, &transpose_order);
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: shape_to_vec(&transposed_reshape_str)?,
        dtype,
        is_const: false,
      },
    );

    let mut shape_with_size = transposed_reshape_str.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpReshape(bytes.to_string(), [node]));
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    *root = Some(node);
  }
  Ok(())
}
