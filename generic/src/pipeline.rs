use std::cell::RefCell;
use std::cmp::min;
use std::env;
use std::path::PathBuf;
use std::process;
use std::rc::Rc;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;

use egg::{Runner, StopReason};

pub use backend::{ir, isel, malloc};
pub use backend::{
    N, PROCESSED, SATURATION_FACTOR, SLEEP_TIME, SLOW_LIMIT_CUTOFF, SLOW_LIMIT_START, TIME_LIMIT,
};

fn print_help(program_name: String) {
    if program_name == "cargo" {
        // When run via `cargo run -- ...`, the first argument is "cargo"
        println!("Usage: cargo run -- --input <hlo_path> --output <asm_path> [--log <log_dir>]");
    } else {
        println!(
            "Usage: {} --input <hlo_path> --output <asm_path> [--log <log_dir>]",
            program_name
        );
    }
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
    cost: i32,
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
                    std::fs::remove_file(output_path)
                        .expect("Failed to remove existing output file");
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

fn update_best(best: &mut Option<AsmCandidate>, asm_path: &PathBuf, start: &Instant) -> bool {
    if !asm_path.exists() {
        return false;
    }

    println!("Current elapsed: {:?} | Will stop at {:?} if no progress", start.elapsed(), TIME_LIMIT);
    println!("Starting Phase X: Performance Cost Evaluation for {:?}", asm_path);
    println!();

    let cost: i32 = backend::cost::python_bridge(&asm_path);
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
                return true;
            } else {
                println!("Not better than the previous best ({}).", best_cost);
                return false;
            }
        }
        None => {
            println!("First ASM found, setting as best.");
            *best = Some(AsmCandidate {
                path: asm_path.clone(),
                cost,
                timestamp: time,
            });
            return true;
        }
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

    let pii_counter: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
    let best: Rc<RefCell<Option<AsmCandidate>>> = Rc::new(RefCell::new(None));

    // Start processing the input file
    let start = Instant::now();

    check_termination(best.borrow().clone(), &start, &output_path);
    println!("Starting Phase 1: Instruction Selection...");
    println!();

    check_termination(best.borrow().clone(), &start, &output_path);
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

    let mut limit: usize = SLOW_LIMIT_START;

    let rules = isel::rewrites::get_rewrites();
    let inputs_for_hook = inputs.clone();
    let hbm_offsets_for_hook = hbm_offsets.clone();

    let output_path_for_hook = output_path.clone();
    let log_dir_for_hook = log_dir.clone();
    let metadata_path_for_hook = metadata_path.clone();

    let runner = {
        // clone the Rcs
        let pii_counter = pii_counter.clone();
        let best = best.clone();

        Runner::default()
            .with_egraph(init_egraph)
            .with_node_limit(5000)
            .with_time_limit(TIME_LIMIT)
            .with_hook(move |runner| {
                PROCESSED.lock().unwrap().clear();
                if runner.iterations.len() % N == 0 && runner.iterations.len() > 0 {
                    check_termination(best.borrow().clone(), &start, &output_path_for_hook);
                    println!(
                        "Starting Phase 1 Module 3: Graph Extractor (limit {})",
                        limit
                    );
                    println!();

                    let piis = isel::extractor::extract(
                        &mut runner.egraph.clone(),
                        root,
                        &inputs_for_hook,
                        &hbm_offsets_for_hook,
                        limit,
                    );
                    limit += 1; // Increment limit to allow for more extraction next time

                    for pii in piis {
                        check_termination(best.borrow().clone(), &start, &output_path_for_hook);
                        println!("Starting Phase 2 for PII #{}", *pii_counter.borrow());
                        println!();

                        // Phase 2
                        let pii_path =
                            log_dir_for_hook.join(format!("{}.pii", *pii_counter.borrow()));
                        pii.save(&pii_path);

                        let asm_path =
                            log_dir_for_hook.join(format!("{}.py", *pii_counter.borrow()));

                        backend::malloc::cpp_bridge(&pii_path, &metadata_path_for_hook, &asm_path);

                        // Check cost and update best if necessary
                        update_best(&mut *best.borrow_mut(), &asm_path, &start);
                        println!();

                        *pii_counter.borrow_mut() += 1;
                    }

                    check_termination(best.borrow().clone(), &start, &output_path_for_hook);
                    println!("Completed Phase 2: Memory Allocation, returning to Phase 1: Instruction Selection");
                    println!();
                }

                check_termination(best.borrow().clone(), &start, &output_path_for_hook);
                println!(
                    "Starting Phase 1 Module 2: Rewrite Applier (iteration {})",
                    runner.iterations.len() + 1
                );
                println!();
                sleep(SLEEP_TIME);

                Ok(())
            })
            .run(&rules.clone())
    };

    // Logic based on the stop reason:
    // 1. TimeLimit: No more extraction, just return.
    // 2. NodeLimit, Saturated: Run extraction until time limit is hit.
    // 3. IterationLimit: Should not have happened. Recheck if there is a default limit.
    // 4. Other: Should not have happened. Requires investigation.

    match runner.stop_reason.as_ref().unwrap() {
        StopReason::TimeLimit(_) => {
            println!("Info: Reached time limit. No further extraction.");
            println!();
        }
        StopReason::NodeLimit(_) | StopReason::Saturated => {
            println!("Info: Reached node limit or saturated. Running extraction until time limit is hit.");
            println!();

            while start.elapsed() < TIME_LIMIT {
                check_termination(best.borrow().clone(), &start, &output_path);
                println!(
                    "Starting Phase 1 Module 3: Graph Extractor (limit {})",
                    limit
                );
                println!();

                let piis = isel::extractor::extract(
                    &mut runner.egraph.clone(),
                    root,
                    &inputs,
                    &hbm_offsets,
                    limit,
                );
                limit += 1; // Increment limit to allow for more extraction next time

                for pii in piis {
                    check_termination(best.borrow().clone(), &start, &output_path);
                    println!("Starting Phase 2 for PII #{}", *pii_counter.borrow());
                    println!();

                    // Phase 2
                    let pii_path = log_dir.join(format!("{}.pii", *pii_counter.borrow()));
                    pii.save(&pii_path);

                    let asm_path = log_dir.join(format!("{}.py", *pii_counter.borrow()));
                    backend::malloc::cpp_bridge(&pii_path, &metadata_path, &asm_path);

                    // Check cost and update best if necessary
                    update_best(&mut *best.borrow_mut(), &asm_path, &start);
                    println!();

                    *pii_counter.borrow_mut() += 1;
                }

                check_termination(best.borrow().clone(), &start, &output_path);
                println!("Completed Phase 2: Memory Allocation, returning to Phase 1: Instruction Selection");
                println!();

                sleep(SLEEP_TIME);
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

    timeout(best.borrow().clone(), &start, &output_path);
}
