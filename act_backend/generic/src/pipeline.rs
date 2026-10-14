use std::cell::{Cell, RefCell};
use std::cmp::min;
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use egg::{EGraph, Id, Runner, StopReason};

use backend::ir::egraph::{TensorInfo, TensorOp};
use backend::isel;
use backend::{ITERATION_LIMIT, NODE_LIMIT, SATURATION_FACTOR, SLOW_LIMIT_START, TIME_LIMIT};

fn print_help(program_name: String) {
  println!(
    "Usage: {} --input <hlo_path> --output <asm_path> [--log <log_dir>]",
    program_name
  );
  println!();
  println!("Description:");
  println!("  This program compiles an .hlo file into an assembly code.");
  println!("  The candidates are kept in the log directory (see --log).");
  println!("  The final assembly code is chosen based on performance cost.");
  println!();
  println!("Options:");
  println!("  --help       Print this help message");
  println!("  --input      Specify the input .hlo file path");
  println!("               (required, must have .hlo extension)");
  println!("  --output     Specify the output assembly file path");
  println!("               (required, will be created/overwritten)");
  println!("  --log        Specify the log directory");
  println!("               (optional, defaults to a new directory under the system's");
  println!("               temporary directory)");
  println!();
}

#[derive(Clone)]
struct AsmCandidate {
  path: PathBuf,
  cost: f64,
  timestamp: Duration,
}

// No assembly: one stderr line and status 1
fn no_assembly(reason: &str) -> ! {
  println!("No output file created.");
  eprintln!("Error: no assembly found: {}", reason);
  process::exit(1);
}

// The output's name until it is whole
fn partial_output(output_path: &Path) -> PathBuf {
  output_path.with_extension(format!("py.{}", process::id()))
}

// Report the search's elapsed time and when it stops
fn report_elapsed(elapsed: Duration, stop: Duration) {
  println!(
    "Current elapsed: {:?} | Will stop at {:?} if no progress",
    elapsed, stop
  );
}

fn update_best(best: &mut Option<AsmCandidate>, asm_path: &PathBuf, start: &Instant) -> bool {
  report_elapsed(start.elapsed(), TIME_LIMIT);
  println!("Starting Performance Cost Evaluation for {:?}", asm_path);
  println!();

  let cost: f64 = match backend::cost::c_bridge(asm_path) {
    Ok(cost) => cost,
    Err(reason) => no_assembly(&format!("the cost model {}", reason)),
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
  first_drop: RefCell<Option<String>>,
  rewrites: RefCell<Option<String>>,
}

impl Search {
  // End the compile once its time limit runs out
  fn check_termination(&self) {
    let current_time = self.start.elapsed();
    if current_time > TIME_LIMIT {
      println!("Reached overall time limit of {:?}, stopping", TIME_LIMIT);
      println!();
      self.finish();
    }
    // Stop once no candidate has improved for long enough
    match self.best.borrow().clone() {
      Some(AsmCandidate { timestamp, .. }) => {
        if current_time > timestamp * SATURATION_FACTOR {
          println!(
            "No improvement for last {:?}, stopping",
            current_time - timestamp
          );
          println!();
          self.finish();
        } else {
          report_elapsed(current_time, min(timestamp * SATURATION_FACTOR, TIME_LIMIT));
        }
      }
      None => {
        report_elapsed(current_time, TIME_LIMIT);
      }
    }
  }

  // End the compile, the best candidate to the output
  fn finish(&self) -> ! {
    println!("Total time: {:?}", self.start.elapsed());
    match self.best.borrow().clone() {
      Some(AsmCandidate {
        path,
        cost,
        timestamp,
      }) => {
        println!(
          "Best ASM {:?} with cost {} found at {:?}",
          path, cost, timestamp
        );

        // The output takes its name once it is whole
        let partial = partial_output(&self.output_path);
        std::fs::copy(&path, &partial).expect("Failed to copy best ASM to output path");
        std::fs::rename(&partial, &self.output_path).expect("Failed to name the output");
        println!(
          "Best ASM copied to output path: {}",
          self.output_path.display()
        );
      }
      None => {
        println!("Could not find an ASM representation.");
        // What became of the candidates
        let candidates = match &*self.first_drop.borrow() {
          Some(reason) => format!("every candidate was dropped, the first because {}", reason),
          None => "no candidate was extracted".to_string(),
        };
        // What ended the search
        let ended = match &*self.rewrites.borrow() {
          Some(rewrites) => format!(
            "{} and the time limit of {:?} ran out",
            rewrites, TIME_LIMIT
          ),
          None => format!("the time limit of {:?} ran out", TIME_LIMIT),
        };
        no_assembly(&format!("{}, and {}", ended, candidates));
      }
    }
    process::exit(0);
  }

  // Extract the e-graph at one limit and evaluate each candidate
  fn pass(&self, egraph: &EGraph<TensorOp, TensorInfo>, limit: usize) {
    self.check_termination();
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
      self.check_termination();
      println!("Starting Phase 2 for PII #{}", number);
      println!();

      // Phase 2
      let pii_path = self.log_dir.join(format!("{}.pii", number));
      pii.save(&pii_path);

      let asm_path = self.log_dir.join(format!("{}.py", number));
      match backend::malloc::cpp_bridge(&pii_path, &self.metadata_path, &asm_path) {
        // Check cost and update best if necessary
        Ok(()) => {
          update_best(&mut self.best.borrow_mut(), &asm_path, &self.start);
        }
        Err(reason) => {
          println!("PII #{} dropped: {}", number, reason);
          self.first_drop.borrow_mut().get_or_insert(reason);
        }
      }
      println!();

      self.candidates.set(number + 1);
    }

    self.check_termination();
    println!("Completed Phase 2: Memory Allocation, returning to Phase 1: Instruction Selection");
    println!();
  }
}

fn main() {
  let args: Vec<String> = env::args().collect();

  if args.contains(&"--help".to_string()) {
    print_help(args[0].clone());
    process::exit(0);
  }

  // A run without a required flag is a usage error
  for flag in ["--input", "--output"] {
    if !args.contains(&flag.to_string()) {
      print_help(args[0].clone());
      eprintln!("Error: missing the required flag {}", flag);
      process::exit(1);
    }
  }

  // Process input file
  let input_index = args.iter().position(|x| x == "--input").unwrap() + 1;
  if input_index >= args.len() {
    eprintln!("Error: missing file name after --input");
    process::exit(1);
  }

  let hlo_path_arg = &args[input_index];
  let current_dir = env::current_dir().expect("Failed to get current directory");
  let hlo_path = current_dir.join(hlo_path_arg);

  if hlo_path.extension().and_then(|s| s.to_str()) != Some("hlo") {
    eprintln!(
      "Error: input file '{}' is not an .hlo file",
      hlo_path.display()
    );
    process::exit(1);
  }

  if !hlo_path.exists() {
    eprintln!("Error: input file '{}' does not exist", hlo_path.display());
    process::exit(1);
  }

  println!("Input file: {}", hlo_path.display());

  // Process output file
  let output_index = args.iter().position(|x| x == "--output").unwrap() + 1;
  if output_index >= args.len() {
    eprintln!("Error: missing file name after --output");
    process::exit(1);
  }
  let output_path_arg = &args[output_index];
  let output_path = current_dir.join(output_path_arg);

  if output_path.extension().and_then(|s| s.to_str()) != Some("py") {
    eprintln!(
      "Error: output file '{}' does not have a .py extension",
      output_path.display()
    );
    process::exit(1);
  }

  println!("Output file: {}", output_path.display());

  // Process log directory
  let own_log = !args.contains(&"--log".to_string());
  let log_dir = if own_log {
    // A directory per process, named by its id
    env::temp_dir().join(format!("act-backend-{}", process::id()))
  } else {
    let log_index = args.iter().position(|x| x == "--log").unwrap() + 1;
    if log_index >= args.len() {
      eprintln!("Error: missing directory after --log");
      process::exit(1);
    }
    let log_dir = PathBuf::from(&args[log_index]);
    // A passed directory is written into, nothing in it removed
    if log_dir.exists() && !log_dir.is_dir() {
      eprintln!(
        "Error: the log directory '{}' is not a directory",
        log_dir.display()
      );
      process::exit(1);
    }
    if let Err(error) = std::fs::create_dir_all(&log_dir) {
      eprintln!(
        "Error: cannot create the log directory '{}': {}",
        log_dir.display(),
        error
      );
      process::exit(1);
    }
    log_dir
  };
  println!("Log directory: {}", log_dir.display());

  // The output's directory is made once every flag is checked
  let output_dir = output_path.parent().unwrap();
  if let Err(error) = std::fs::create_dir_all(output_dir) {
    eprintln!(
      "Error: cannot create the output's directory '{}': {}",
      output_dir.display(),
      error
    );
    process::exit(1);
  }

  // An earlier compile's output goes before this one starts
  if output_path.exists() {
    if let Err(error) = std::fs::remove_file(&output_path) {
      eprintln!(
        "Error: cannot remove the earlier output '{}': {}",
        output_path.display(),
        error
      );
      process::exit(1);
    }
  }

  // The output's directory takes a file before the search starts
  let partial = partial_output(&output_path);
  if let Err(error) = std::fs::File::create(&partial).and_then(|_| std::fs::remove_file(&partial)) {
    eprintln!(
      "Error: cannot write the output '{}': {}",
      output_path.display(),
      error
    );
    process::exit(1);
  }

  // Remove a leftover, then make the compile's own log directory
  if own_log {
    if log_dir.exists() {
      std::fs::remove_dir_all(&log_dir).expect("Failed to remove existing log directory");
    }
    if let Err(error) = std::fs::create_dir_all(&log_dir) {
      eprintln!(
        "Error: cannot create the log directory '{}': {}",
        log_dir.display(),
        error
      );
      process::exit(1);
    }
  }

  println!("PII graphs dumped to: {}", log_dir.display());
  println!();

  // Start processing the input file
  let start = Instant::now();

  report_elapsed(start.elapsed(), TIME_LIMIT);
  println!("Starting Phase 1: Instruction Selection...");
  println!();

  report_elapsed(start.elapsed(), TIME_LIMIT);
  println!("Starting Phase 1 Module 1: E-Graph Initializer...");
  println!();

  let (init_egraph, hbm_offsets, root, inputs, mut metadata) =
    match isel::initializer::parse_hlo_module_to_egraph(&hlo_path) {
      Ok(parsed) => parsed,
      Err(reason) => no_assembly(&reason),
    };

  println!("HBM Offsets: {:?}", hbm_offsets);
  println!("Root ID: {:?}", root);
  println!("Inputs: {:?}", inputs);
  println!();

  let metadata_path = log_dir.join("metadata.json");
  // The kernel's path as given, control characters escaped
  metadata.input_file = hlo_path_arg
    .chars()
    .map(|c| {
      if c.is_control() {
        c.escape_default().to_string()
      } else {
        c.to_string()
      }
    })
    .collect();
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
    first_drop: RefCell::new(None),
    rewrites: RefCell::new(None),
  });

  let rules = isel::rewrites::get_rewrites();

  let runner = {
    let search = search.clone();
    let mut limit: usize = SLOW_LIMIT_START;

    Runner::default()
      .with_egraph(init_egraph)
      .with_node_limit(NODE_LIMIT)
      .with_iter_limit(ITERATION_LIMIT)
      .with_time_limit(TIME_LIMIT)
      .with_hook(move |runner| {
        // Extract after every iteration
        let extract_now = !runner.iterations.is_empty();
        if extract_now {
          search.pass(&runner.egraph, limit);
          limit += 1; // Increment limit to allow for more extraction next time
        }

        search.check_termination();
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
    StopReason::NodeLimit(_) | StopReason::IterationLimit(_) | StopReason::Saturated => {
      // What ended the rewrites, which the no-assembly line names
      let rewrites = match runner.stop_reason {
        Some(StopReason::NodeLimit(_)) => {
          format!("the rewrites reached the node limit of {}", NODE_LIMIT)
        }
        Some(StopReason::IterationLimit(_)) => format!(
          "the rewrites reached the iteration limit of {}",
          ITERATION_LIMIT
        ),
        _ => "the rewrites saturated".to_string(),
      };
      println!(
        "Info: {}. Running extraction until time limit is hit.",
        rewrites
      );
      println!();
      *search.rewrites.borrow_mut() = Some(rewrites);

      let mut limit: usize = SLOW_LIMIT_START;
      while start.elapsed() < TIME_LIMIT {
        search.pass(&runner.egraph, limit);
        limit += 1; // Increment limit to allow for more extraction next time
      }
      println!("Info: Reached time limit. No further extraction.");
    }
    StopReason::Other(_) => {
      eprintln!("Error: Stopped for an unexpected reason. Requires investigation.");
      process::exit(1);
    }
  }

  println!("Reached overall time limit of {:?}, stopping", TIME_LIMIT);
  println!();
  search.finish();
}
