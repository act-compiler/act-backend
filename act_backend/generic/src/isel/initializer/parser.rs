use crate::isel::initializer::expression::{parse_hlo_expr, HLOExpr};
use crate::isel::initializer::line::{other_than_row_major, parse_line};

#[derive(Debug, Clone)]
pub struct HLORegion {
  pub name: String,
  pub instructions: Vec<HLOExpr>,
}

#[derive(Debug)]
pub struct HloModuleHeader {
  pub module_name: String,
  pub input_types: Vec<String>,
  pub output_type: String,
}

pub fn parse_hlo_module_header(line: &str) -> Result<HloModuleHeader, String> {
  let line = line.trim();
  // Why the reader cannot split a header
  let malformed = || {
    format!(
      "the kernel's first line '{}' is not an HloModule line with an entry_computation_layout",
      line
    )
  };

  // Step 1: Extract module name and entry computation layout
  let prefix = "HloModule ";
  if !line.starts_with(prefix) {
    return Err(malformed());
  }
  let rest = &line[prefix.len()..];

  // Get module name (first part before comma)
  let module_name = rest.split(',').next().unwrap_or("").trim().to_string();

  // Find entry_computation_layout parameter
  let layout_start = "entry_computation_layout={";
  let start_pos = rest.find(layout_start).ok_or_else(malformed)?;
  let layout_part = &rest[start_pos + layout_start.len()..];

  // Find matching closing brace
  let mut brace_count = 1;
  let mut layout_end = 0;
  for (i, ch) in layout_part.char_indices() {
    match ch {
      '{' => brace_count += 1,
      '}' => {
        brace_count -= 1;
        if brace_count == 0 {
          layout_end = i;
          break;
        }
      }
      _ => {}
    }
  }

  if brace_count != 0 {
    return Err(malformed());
  }

  let entry_layout = &layout_part[..layout_end];

  // Step 2: Remove layout specifications like {1,0}
  let mut layout_cleaned = entry_layout.to_string();
  while let Some(start) = layout_cleaned.find('{') {
    if let Some(end) = layout_cleaned[start..].find('}') {
      let end_pos = start + end;
      // The shape the layout follows
      if let Some(shape_start) = layout_cleaned[..start].rfind('[') {
        let shape = layout_cleaned[shape_start + 1..start].trim_end_matches(']');
        if let Some(reason) = other_than_row_major(shape, &layout_cleaned[start + 1..end_pos]) {
          return Err(reason);
        }
      }
      layout_cleaned = format!(
        "{}{}",
        &layout_cleaned[..start],
        &layout_cleaned[end_pos + 1..]
      );
    } else {
      break;
    }
  }

  // Step 3: Process the cleaned layout
  let layout_no_spaces = layout_cleaned.replace(" ", "");

  // Assume format: (input_types) -> output_type (no parentheses around output)
  let mut normalized_layout = layout_no_spaces.clone();
  if let Some(close_paren_pos) = normalized_layout.find(')') {
    if let Some(arrow_pos) = normalized_layout[close_paren_pos..].find("->") {
      let actual_arrow_pos = close_paren_pos + arrow_pos;
      normalized_layout = format!(
        "{}|{}",
        &normalized_layout[..close_paren_pos],
        &normalized_layout[actual_arrow_pos + 2..] // Skip the "->"
      );
    } else {
      return Err(malformed());
    }
  } else {
    return Err(malformed());
  }

  let io_parts: Vec<&str> = normalized_layout.split('|').collect();
  if io_parts.len() != 2 {
    return Err(malformed());
  }

  // Extract inputs and output
  let inputs_str = io_parts[0].strip_prefix('(').unwrap_or(io_parts[0]);
  let output_type = io_parts[1].strip_suffix(')').unwrap_or(io_parts[1]);

  // Split inputs on top-level commas (ignore commas inside brackets)
  fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut depth = 0usize; // bracket depth for '[' and ']'

    for ch in s.chars() {
      match ch {
        '[' => {
          depth += 1;
          buf.push(ch);
        }
        ']' => {
          depth = depth.saturating_sub(1);
          buf.push(ch);
        }
        ',' if depth == 0 => {
          let piece = buf.trim();
          if !piece.is_empty() {
            parts.push(piece.to_string());
          }
          buf.clear();
        }
        _ => buf.push(ch),
      }
    }
    let piece = buf.trim();
    if !piece.is_empty() {
      parts.push(piece.to_string());
    }
    parts
  }

  let input_types = if inputs_str.is_empty() {
    Vec::new()
  } else {
    split_top_level_commas(inputs_str)
  };

  Ok(HloModuleHeader {
    module_name,
    input_types,
    output_type: output_type.to_string(),
  })
}

pub fn parse_region(
  lines: &[&str],
  start_idx: usize,
  regions: &std::collections::HashMap<String, HLORegion>,
) -> Result<Option<(HLORegion, usize)>, String> {
  if start_idx >= lines.len() {
    return Ok(None);
  }

  let first_line = lines[start_idx].trim();

  // A region opens with its name and a brace
  let Some(brace_pos) = first_line.find('{') else {
    return Ok(None);
  };

  let region_name_with_signature = first_line[..brace_pos].trim();

  // The region's name, before its signature: %name (params) -> type
  let region_name = if let Some(paren_pos) = region_name_with_signature.find('(') {
    region_name_with_signature[..paren_pos].trim().to_string()
  } else {
    region_name_with_signature.to_string()
  };
  let mut instructions = Vec::new();
  let mut idx = start_idx + 1;
  let mut brace_count = 1; // The opening brace, found above

  // Parse up to the matching closing brace
  while idx < lines.len() && brace_count > 0 {
    let line = lines[idx].trim();

    // Count braces to handle nested regions
    for ch in line.chars() {
      match ch {
        '{' => brace_count += 1,
        '}' => brace_count -= 1,
        _ => {}
      }
    }

    // Parse a line that holds more than braces
    if !line.is_empty()
      && !line
        .chars()
        .all(|c| c == '{' || c == '}' || c.is_whitespace())
    {
      // Remove the closing brace if it's on this line
      let instruction_line = if line.ends_with('}') && brace_count == 0 {
        line.strip_suffix('}').unwrap_or(line).trim()
      } else {
        line
      };

      if !instruction_line.is_empty() {
        if let Some(((tsymbol, ttype, tshape, ttile, toperator, toperands, trest), _)) =
          parse_line(instruction_line)?
        {
          if let Some(expr) = parse_hlo_expr(
            &tsymbol, &ttype, &tshape, &ttile, &toperator, &toperands, &trest, regions,
          )? {
            instructions.push(expr);
          }
        }
      }
    }

    idx += 1;
  }

  Ok(Some((
    HLORegion {
      name: region_name,
      instructions,
    },
    idx,
  )))
}

pub fn parse_hlo_module_with_regions(
  lines: &[&str],
) -> Result<(HloModuleHeader, Vec<HLOExpr>, usize), String> {
  if lines.is_empty() {
    return Err("the kernel is empty".to_string());
  }
  // Parse the module header from the first line
  let header = parse_hlo_module_header(lines[0])?;

  let mut regions = std::collections::HashMap::new();
  let mut entry_instructions = Vec::new();
  // The ENTRY's ROOT lines, by their place among its instructions
  let mut roots = vec![];
  let mut idx = 1;

  // Parse regions and entry computation
  while idx < lines.len() {
    let line = lines[idx].trim();

    if line.is_empty() {
      idx += 1;
      continue;
    }

    if line.starts_with("ENTRY") {
      // Parse entry computation
      idx += 1; // Skip the ENTRY line
      while idx < lines.len() {
        let instruction_line = lines[idx].trim();
        if instruction_line.is_empty() {
          idx += 1;
          continue;
        }

        if instruction_line == "}" {
          break; // End of entry computation
        }

        if let Some(((tsymbol, ttype, tshape, ttile, toperator, toperands, trest), is_root)) =
          parse_line(instruction_line)?
        {
          if let Some(expr) = parse_hlo_expr(
            &tsymbol, &ttype, &tshape, &ttile, &toperator, &toperands, &trest, &regions,
          )? {
            if is_root {
              roots.push(entry_instructions.len());
            }
            entry_instructions.push(expr);
          }
        }
        idx += 1;
      }
    } else if line.contains('{') {
      // This is a region definition
      if let Some((region, new_idx)) = parse_region(lines, idx, &regions)? {
        regions.insert(region.name.clone(), region);
        idx = new_idx;
      } else {
        idx += 1;
      }
    } else {
      idx += 1;
    }
  }

  match roots[..] {
    [root] => Ok((header, entry_instructions, root)),
    // With no line marked, the last line is the root
    [] => {
      let last = entry_instructions.len().saturating_sub(1);
      Ok((header, entry_instructions, last))
    }
    _ => Err(format!(
      "the kernel's ENTRY marks {} lines ROOT, where the reader takes one",
      roots.len()
    )),
  }
}
