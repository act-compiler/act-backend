pub(super) fn shape_to_vec(shape_str: &str) -> Result<Vec<i32>, String> {
  let cleaned = shape_str.trim();

  if cleaned.is_empty() || cleaned == "[]" {
    return Ok(vec![1]);
  }

  let trimmed = cleaned.trim_matches(|c| c == '[' || c == ']');

  if trimmed.trim().is_empty() {
    return Ok(vec![1]);
  }

  trimmed
    .split(',')
    .filter(|s| !s.trim().is_empty())
    .map(|x| dimension(x, shape_str))
    .collect()
}

// One dimension of a shape, or why it is none
fn dimension(text: &str, shape: &str) -> Result<i32, String> {
  text.trim().parse::<i32>().map_err(|_| {
    format!(
      "the shape [{}] holds '{}', which is not a dimension",
      shape.trim_matches(|c| c == '[' || c == ']'),
      text.trim()
    )
  })
}

// A shape's dimensions as a kernel writes them
pub(super) fn shape_text(shape: &[i32]) -> String {
  shape
    .iter()
    .map(|size| size.to_string())
    .collect::<Vec<_>>()
    .join(",")
}

// The assembly's declared shape, a rank-0 value as ()
pub(super) fn declared_shape(tshape: &str) -> Result<Vec<i32>, String> {
  if tshape == "[]" {
    Ok(vec![])
  } else {
    shape_to_vec(tshape)
  }
}

// A shape plus a dimension for a multi-byte element's bytes
pub(super) fn byte_view(tshape: &str, dtype_size: i32) -> String {
  // A rank-0 value is held as one element
  let shape = if tshape == "[]" { "1" } else { tshape };
  if dtype_size == 1 {
    shape.to_string()
  } else {
    format!("{},{}", shape, dtype_size)
  }
}

pub(super) fn get_bytes(shape: &str, dtype: i32) -> Result<i32, String> {
  let mut bytes = 1;
  shape_to_vec(shape)?.iter().for_each(|dim| bytes *= dim);
  Ok(bytes * dtype)
}

pub(super) fn get_tiled_shape(shape: &str, tile: &str) -> Result<String, String> {
  let shape_vec = shape
    .split(',')
    .map(|s| dimension(s, shape))
    .collect::<Result<Vec<i32>, String>>()?;
  // A tiled shape's dimensions are never negative
  if let Some(dim) = shape_vec.iter().find(|dim| **dim < 0) {
    return Err(format!(
      "the shape [{}] holds '{}', which is not a dimension",
      shape, dim
    ));
  }

  let tile_vec = tile
    .split(',')
    .map(|s| tile_size(s, shape, tile))
    .collect::<Result<Vec<i32>, String>>()?;

  if shape_vec.len() != tile_vec.len() {
    return Err(format!(
      "the tile ({}) of the shape [{}] does not give one size per dimension",
      tile, shape
    ));
  }

  let divided: Vec<String> = shape_vec
    .iter()
    .zip(tile_vec.iter())
    .map(|(s, t)| (s / t).to_string())
    .collect();

  let mut result = divided.join(",");
  result.push(',');
  result.push_str(
    &tile_vec
      .iter()
      .map(|t| t.to_string())
      .collect::<Vec<_>>()
      .join(","),
  );

  Ok(result)
}

// One size of a tile, or why it is none
fn tile_size(text: &str, shape: &str, tile: &str) -> Result<i32, String> {
  match text.trim().parse::<i32>() {
    Ok(size) if size > 0 => Ok(size),
    _ => Err(format!(
      "the tile ({}) of the shape [{}] holds '{}', which is not a size",
      tile,
      shape,
      text.trim()
    )),
  }
}

pub(super) fn get_result_shape(tshape: &str, ttile: &str) -> Result<String, String> {
  let dims = tshape
    .split(',')
    .map(|x| dimension(x, tshape))
    .collect::<Result<Vec<i32>, String>>()?;
  let tiles = ttile
    .split(',')
    .map(|x| tile_size(x, tshape, ttile))
    .collect::<Result<Vec<i32>, String>>()?;

  if dims.len() != tiles.len() {
    return Err(format!(
      "the tile ({}) of the shape [{}] does not give one size per dimension",
      ttile, tshape
    ));
  }

  let mut result = Vec::new();
  for (d, t) in dims.iter().zip(tiles.iter()) {
    result.push((d / t).to_string()); // integer division
    result.push(t.to_string());
  }

  Ok(result.join(","))
}

pub(super) fn get_transpose_order_param(num_dims: i32) -> String {
  let mut order = Vec::with_capacity((num_dims * 2) as usize);
  for i in 0..num_dims {
    order.push(i + 1); // outer dimension (1-based)
    order.push(i + 1 + num_dims); // tile dimension (1-based + offset)
  }

  order
    .iter()
    .map(|x| x.to_string())
    .collect::<Vec<String>>()
    .join(",")
}

pub(super) fn get_transpose_order_result(n: i32) -> String {
  let mut odds = (1..=2 * n).step_by(2).collect::<Vec<_>>();
  let evens = (2..=2 * n).step_by(2).collect::<Vec<_>>();

  odds.extend(evens);

  odds
    .iter()
    .map(|x| x.to_string())
    .collect::<Vec<_>>()
    .join(",")
}

pub(super) fn apply_transpose_order(reshape_str: &str, order_str: &str) -> String {
  let dims: Vec<&str> = reshape_str.split(',').map(|x| x.trim()).collect();
  let order: Vec<usize> = order_str
    .split(',')
    .map(|x| x.trim().parse::<usize>().unwrap())
    .collect();

  let new_dims: Vec<&str> = order.iter().map(|&i| dims[i - 1]).collect();
  new_dims.join(",")
}
