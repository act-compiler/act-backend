pub fn shape_to_vec(shape_str: &str) -> Result<Vec<i32>, String> {
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
pub fn shape_text(shape: &[i32]) -> String {
  shape
    .iter()
    .map(|size| size.to_string())
    .collect::<Vec<_>>()
    .join(",")
}

// The assembly's declared shape, a rank-0 value as ()
pub fn declared_shape(tshape: &str) -> Result<Vec<i32>, String> {
  if tshape == "[]" {
    Ok(vec![])
  } else {
    shape_to_vec(tshape)
  }
}

// A shape plus a dimension for a multi-byte element's bytes
pub fn byte_view(tshape: &str, dtype_size: i32) -> String {
  // A rank-0 value is held as one element
  let shape = if tshape == "[]" { "1" } else { tshape };
  if dtype_size == 1 {
    shape.to_string()
  } else {
    format!("{},{}", shape, dtype_size)
  }
}

pub fn get_bytes(shape: &str, dtype: i32) -> Result<i32, String> {
  let mut bytes = 1;
  shape_to_vec(shape)?.iter().for_each(|dim| bytes *= dim);
  Ok(bytes * dtype)
}

pub fn get_tiled_shape(shape: &str, tile: &str) -> Result<String, String> {
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
  whole_tiles(&shape_vec, &tile_vec, shape, tile)?;

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

// Whether each dimension holds whole tiles, or why not
fn whole_tiles(dims: &[i32], tiles: &[i32], shape: &str, tile: &str) -> Result<(), String> {
  if dims.iter().zip(tiles).any(|(dim, size)| dim % size != 0) {
    return Err(format!(
      "the tile ({}) does not divide the shape [{}]",
      tile, shape
    ));
  }
  Ok(())
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

pub fn get_result_shape(tshape: &str, ttile: &str) -> Result<String, String> {
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
  whole_tiles(&dims, &tiles, tshape, ttile)?;

  let mut result = Vec::new();
  for (d, t) in dims.iter().zip(tiles.iter()) {
    result.push((d / t).to_string()); // integer division
    result.push(t.to_string());
  }

  Ok(result.join(","))
}

pub fn get_transpose_order_param(num_dims: i32) -> String {
  let mut order = Vec::with_capacity((num_dims * 2) as usize);
  for i in 0..num_dims {
    order.push(i); // outer dimension
    order.push(i + num_dims); // its tile dimension
  }

  order
    .iter()
    .map(|x| x.to_string())
    .collect::<Vec<String>>()
    .join(",")
}

pub fn get_transpose_order_result(n: i32) -> String {
  // Outer dimensions first, then the tile dimensions
  let mut outer = (0..2 * n).step_by(2).collect::<Vec<_>>();
  let tiles = (1..2 * n).step_by(2).collect::<Vec<_>>();

  outer.extend(tiles);

  outer
    .iter()
    .map(|x| x.to_string())
    .collect::<Vec<_>>()
    .join(",")
}

pub fn apply_transpose_order(reshape_str: &str, order_str: &str) -> String {
  let dims: Vec<&str> = reshape_str.split(',').map(|x| x.trim()).collect();
  let order: Vec<usize> = order_str
    .split(',')
    .map(|x| x.trim().parse::<usize>().unwrap())
    .collect();

  let new_dims: Vec<&str> = order.iter().map(|&i| dims[i]).collect();
  new_dims.join(",")
}
