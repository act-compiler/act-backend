pub(super) fn shape_to_vec(shape_str: &str) -> Vec<i32> {
  let cleaned = shape_str.trim();

  if cleaned.is_empty() || cleaned == "[]" {
    return vec![1];
  }

  let trimmed = cleaned.trim_matches(|c| c == '[' || c == ']');

  if trimmed.trim().is_empty() {
    return vec![1];
  }

  trimmed
    .split(',')
    .filter(|s| !s.trim().is_empty())
    .map(|x| x.trim().parse::<i32>().expect("Invalid shape dimension"))
    .collect()
}

pub(super) fn get_bytes(shape: &str, dtype: i32) -> i32 {
  let mut bytes = 1;
  shape_to_vec(shape).iter().for_each(|dim| bytes *= dim);
  bytes * dtype
}

pub(super) fn get_tiled_shape(shape: &str, tile: &str) -> String {
  let shape_vec: Vec<u32> = shape
    .split(',')
    .map(|s| s.trim().parse().expect("Invalid shape number"))
    .collect();

  let tile_vec: Vec<u32> = tile
    .split(',')
    .map(|s| s.trim().parse().expect("Invalid tile number"))
    .collect();

  if shape_vec.len() != tile_vec.len() {
    panic!("Shape and tile must have the same number of dimensions");
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

  result
}

pub(super) fn get_result_shape(tshape: &str, ttile: &str) -> String {
  let dims: Vec<i32> = tshape
    .split(',')
    .map(|x| x.trim().parse::<i32>().unwrap())
    .collect();
  let tiles: Vec<i32> = ttile
    .split(',')
    .map(|x| x.trim().parse::<i32>().unwrap())
    .collect();

  assert_eq!(dims.len(), tiles.len(), "Shape and tile length mismatch");

  let mut result = Vec::new();
  for (d, t) in dims.iter().zip(tiles.iter()) {
    result.push((d / t).to_string()); // integer division
    result.push(t.to_string());
  }

  result.join(",")
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
