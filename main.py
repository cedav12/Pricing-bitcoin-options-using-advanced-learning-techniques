import argparse
from src.dataset_builder import DatasetBuilder
from src.black_scholes import BlackScholesBenchmark

def main():
    parser = argparse.ArgumentParser(description="Bitcoin Option Pricing Project")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["build_dataset", "black_scholes"],
        help="Execution mode: 'build_dataset' to process raw data, 'black_scholes' to run BS benchmark."
    )
    
    args = parser.parse_args()
    
    if args.mode == "build_dataset":
        print("Starting dataset preparation pipeline...")
        builder = DatasetBuilder()
        builder.build_dataset()
        print("Dataset preparation complete.")
        
    elif args.mode == "black_scholes":
        print("Running Black-Scholes benchmark...")
        benchmark = BlackScholesBenchmark()
        # You can adjust default risk-free rate and volatility here or pass them as args later.
        benchmark.run_benchmark(r=0.0, sigma=0.8)
        print("Black-Scholes benchmark complete.")

if __name__ == "__main__":
    main()
