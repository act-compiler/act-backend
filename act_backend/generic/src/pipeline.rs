use std::cell::{Cell, RefCell};
use std::cmp::min;
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use egg::{EGraph, Id, Runner, StopReason};

use backend::ir::egraph::{TensorInfo, TensorOp};
use backend::isel;
use backend::{SATURATION_FACTOR, SLOW_LIMIT_START, TIME_LIMIT};

fn print_help(program_name: String) {
  println!(
    "Usage: {} --input <hlo_path> --output <asm_path> [--log <log_dir>]",
    program_name
  );
  println!();
  println!("Description:");
  println!("  This program compiles an .hlo file into an assembly code.");
  println!("  Candidate assembly codes are logged in the specified log directory.");
  println!("  The final assembly code is chosen based on performance cost.");
  println!();
  println!("Options:");
  println!("  --help       Print this help message");
  println!("  --input      Specify the input .hlo file path");
  println!("               (required, must have .hlo extension)");
  println!("  --output     Specify the output assembly file path");
  println!("               (required, will be created/overwritten)");
  println!("  --log        Specify the log directory");
  println!("               (optional, defaults to /tmp/log if not provided)");
  println!();
}

#[derive(Clone)]
struct AsmCandidate {
  path: PathBuf,
  cost: f64,
  timestamp: Duration,
}

fn check_termination(best: Option<AsmCandidate>, start: &Instant, output_path: &PathBuf) {
  let current_time = start.elapsed();
  match best {
    Some(AsmCandidate {
      path,
      cost,
      timestamp,
    }) => {
      if current_time > timestamp * SATURATION_FACTOR {
        println!(
          "No improvement for last {:?}, stopping",
          current_time - timestamp
        );
        println!();
        println!("Total time: {:?}", current_time);
        println!(
          "Best ASM {:?} with cost {} found at {:?}",
          path, cost, timestamp
        );

        std::fs::copy(&path, output_path).expect("Failed to copy best ASM to output path");
        println!("Best ASM copied to output path: {}", output_path.display());

        process::exit(0);
      } else {
        println!(
          "Current elapsed: {:?} | Will stop at {:?} if no progress",
          current_time,
          min(timestamp * SATURATION_FACTOR, TIME_LIMIT)
        );
      }
    }
    None => {
      println!(
        "Current elapsed: {:?} | Will stop at {:?} if no progress",
        current_time, TIME_LIMIT
      );
    }
  }
}

fn timeout(best: Option<AsmCandidate>, start: &Instant, output_path: &PathBuf) {
  let current_time = start.elapsed();
  if current_time > TIME_LIMIT {
    println!("Reached overall time limit of {:?}, stopping", TIME_LIMIT);
    println!();
    println!("Total time: {:?}", current_time);
    match best {
      Some(AsmCandidate {
        path,
        cost,
        timestamp,
      }) => {
        println!(
          "Best ASM {:?} with cost {} found at {:?}",
          path, cost, timestamp
        );

        std::fs::copy(&path, output_path).expect("Failed to copy best ASM to output path");
        println!("Best ASM copied to output path: {}", output_path.display());
      }
      None => {
        println!("Could not find an ASM representation.");

        if output_path.exists() {
          std::fs::remove_file(output_path).expect("Failed to remove existing output file");
        }
        println!("No output file created.");
      }
    }
    process::exit(0);
  } else {
    eprintln!(
      "Error: timeout() called at {:?} but overall time limit of {:?} not reached",
      current_time, TIME_LIMIT
    );
    process::exit(1);
  }
}

fn update_best(
  best: &mut Option<AsmCandidate>,
  asm_path: &PathBuf,
  start: &Instant,
  output_path: &PathBuf,
) -> bool {
  if !asm_path.exists() {
    return false;
  }

  println!(
    "Current elapsed: {:?} | Will stop at {:?} if no progress",
    start.elapsed(),
    TIME_LIMIT
  );
  println!("Starting Performance Cost Evaluation for {:?}", asm_path);
  println!();

  let cost: f64 = match backend::cost::c_bridge(asm_path) {
    Ok(cost) => cost,
    Err(reason) => {
      eprintln!("Error: the cost model {} for {:?}", reason, asm_path);

      // A failed compile leaves no output
      if output_path.exists() {
        std::fs::remove_file(output_path).expect("Failed to remove existing output file");
      }
      process::exit(1);
    }
  };
  let time = start.elapsed();

  print!("New ASM has performance cost {}. ", cost);

  match best {
    Some(AsmCandidate {
      cost: best_cost, ..
    }) => {
      if cost < *best_cost {
        println!("Better than the previous best ({}), updating.", best_cost);
        *best = Some(AsmCandidate {
          path: asm_path.clone(),
          cost,
          timestamp: time,
        });
        true
      } else {
        println!("Not better than the previous best ({}).", best_cost);
        false
      }
    }
    None => {
      println!("First ASM found, setting as best.");
      *best = Some(AsmCandidate {
        path: asm_path.clone(),
        cost,
        timestamp: time,
      });
      true
    }
  }
}

// What every pass of the search shares
struct Search {
  start: Instant,
  output_path: PathBuf,
  log_dir: PathBuf,
  metadata_path: PathBuf,
  root: Id,
  inputs: HashSet<Id>,
  hbm_offsets: Vec<(Option<Id>, i32)>,
  best: RefCell<Option<AsmCandidate>>,
  candidates: Cell<usize>,
}

impl Search {
  // Extract the e-graph at one limit and evaluate each candidate
  fn pass(&self, egraph: &EGraph<TensorOp, TensorInfo>, limit: usize) {
    check_termination(self.best.borrow().clone(), &self.start, &self.output_path);
    println!(
      "Starting Phase 1 Module 3: Graph Extractor (limit {})",
      limit
    );
    println!();

    let piis = isel::extractor::extract(
      &mut egraph.clone(),
      self.root,
      &self.inputs,
      &self.hbm_offsets,
      limit,
    );

    for pii in piis {
      let number = self.candidates.get();
      check_termination(self.best.borrow().clone(), &self.start, &self.output_path);
      println!("Starting Phase 2 for PII #{}", number);
      println!();

      // Phase 2
      let pii_path = self.log_dir.join(format!("{}.pii", number));
      pii.save(&pii_path);

      let asm_path = self.log_dir.join(format!("{}.py", number));
      backend::malloc::cpp_bridge(&pii_path, &self.metadata_path, &asm_path);

      // Check cost and update best if necessary
      update_best(
        &mut self.best.borrow_mut(),
        &asm_path,
        &self.start,
        &self.output_path,
      );
      println!();

      self.candidates.set(number + 1);
    }

    check_termination(self.best.borrow().clone(), &self.start, &self.output_path);
    println!("Completed Phase 2: Memory Allocation, returning to Phase 1: Instruction Selection");
    println!();
  }
}

fn main() {
  let args: Vec<String> = env::args().collect();

  if args.contains(&"--help".to_string())
    || !args.contains(&"--input".to_string())
    || !args.contains(&"--output".to_string())
  {
    print_help(args[0].clone());
    process::exit(0);
  }

  // Process input file
  let input_index = args.iter().position(|x| x == "--input").unwrap() + 1;
  if input_index >= args.len() {
    eprintln!("Error: Missing file name after --input");
    process::exit(1);
  }

  let hlo_path_arg = &args[input_index];
  let current_dir = env::current_dir().expect("Failed to get current directory");
  let hlo_path = current_dir.join(hlo_path_arg);

  if hlo_path.extension().and_then(|s| s.to_str()) != Some("hlo") {
    eprintln!(
      "Error: Input file '{}' is not an .hlo file.",
      hlo_path.display()
    );
    process::exit(1);
  }

  if !hlo_path.exists() {
    eprintln!("Error: Input file '{}' does not exist.", hlo_path.display());
    process::exit(1);
  }

  println!("Input file: {}", hlo_path.display());

  // Process output file
  let output_index = args.iter().position(|x| x == "--output").unwrap() + 1;
  if output_index >= args.len() {
    eprintln!("Error: Missing file name after --output");
    process::exit(1);
  }
  let output_path_arg = &args[output_index];
  let output_path = current_dir.join(output_path_arg);

  if output_path.extension().and_then(|s| s.to_str()) != Some("py") {
    eprintln!(
      "Error: Output file '{}' does not have a .py extension.",
      output_path.display()
    );
    process::exit(1);
  }

  if output_path.exists() {
    println!(
      "Warning: Output file '{}' already exists and will be overwritten.",
      output_path.display()
    );
  }

  println!("Output file: {}", output_path.display());

  // Process log directory
  let log_dir_arg: String = if args.contains(&"--log".to_string()) {
    let log_index = args.iter().position(|x| x == "--log").unwrap() + 1;
    if log_index >= args.len() {
      eprintln!("Error: Missing directory after --log");
      process::exit(1);
    }
    args[log_index].clone()
  } else {
    "/tmp/log".to_string()
  };
  println!("Log directory: {}", log_dir_arg);

  let log_dir = std::path::PathBuf::from(log_dir_arg);
  if log_dir.exists() {
    std::fs::remove_dir_all(&log_dir).expect("Failed to remove existing log directory");
  }
  std::fs::create_dir_all(&log_dir).expect("Failed to create log directory");

  println!("PII graphs dumped to: {}", log_dir.display());
  println!();

  // Start processing the input file
  let start = Instant::now();

  check_termination(None, &start, &output_path);
  println!("Starting Phase 1: Instruction Selection...");
  println!();

  check_termination(None, &start, &output_path);
  println!("Starting Phase 1 Module 1: E-Graph Initializer...");
  println!();

  let (init_egraph, hbm_offsets, root, inputs, metadata) =
    isel::initializer::parse_hlo_module_to_egraph(&hlo_path).unwrap();

  println!("HBM Offsets: {:?}", hbm_offsets);
  println!("Root ID: {:?}", root);
  println!("Inputs: {:?}", inputs);
  println!();

  let metadata_path = log_dir.join("metadata.json");
  metadata.save(&metadata_path);

  let search = Rc::new(Search {
    start,
    output_path,
    log_dir,
    metadata_path,
    root,
    inputs,
    hbm_offsets,
    best: RefCell::new(None),
    candidates: Cell::new(0),
  });

  let rules = isel::rewrites::get_rewrites();

  let runner = {
    let search = search.clone();
    let mut limit: usize = SLOW_LIMIT_START;

    Runner::default()
      .with_egraph(init_egraph)
      .with_node_limit(5000)
      .with_time_limit(TIME_LIMIT)
      .with_hook(move |runner| {
        // Extract after every iteration
        let extract_now = !runner.iterations.is_empty();
        if extract_now {
          search.pass(&runner.egraph, limit);
          limit += 1; // Increment limit to allow for more extraction next time
        }

        check_termination(search.best.borrow().clone(), &start, &search.output_path);
        println!(
          "Starting Phase 1 Module 2: Rewrite Applier (iteration {})",
          runner.iterations.len() + 1
        );
        println!();

        Ok(())
      })
      .run(&rules.clone())
  };

  // What follows the search, by its stop reason

  match runner.stop_reason.as_ref().unwrap() {
    StopReason::TimeLimit(_) => {
      println!("Info: Reached time limit. No further extraction.");
      println!();
    }
    StopReason::NodeLimit(_) | StopReason::Saturated => {
      println!(
        "Info: Reached node limit or saturated. Running extraction until time limit is hit."
      );
      println!();

      let mut limit: usize = SLOW_LIMIT_START;
      while start.elapsed() < TIME_LIMIT {
        search.pass(&runner.egraph, limit);
        limit += 1; // Increment limit to allow for more extraction next time
      }
      println!("Info: Reached time limit. No further extraction.");
    }
    StopReason::IterationLimit(_) => {
      eprintln!("Error: Reached iteration limit. This should not happen.");
      process::exit(1);
    }
    StopReason::Other(_) => {
      eprintln!("Error: Stopped for an unexpected reason. Requires investigation.");
      process::exit(1);
    }
  }

  timeout(search.best.borrow().clone(), &start, &search.output_path);
}
