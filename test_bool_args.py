#!/usr/bin/env python
"""
Test script to demonstrate the problem with type=bool in argparse
Run with different arguments to see how argparse handles boolean flags:

Examples:
  python test_bool_args.py --type-bool-flag False
  python test_bool_args.py --type-bool-flag True
  python test_bool_args.py --store-true-flag
  python test_bool_args.py --no-flag
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Demonstrate boolean flag issues in argparse')
    
    # Problematic way: using type=bool (NEVER DO THIS)
    parser.add_argument('--type-bool-flag', type=bool, default=True,
                      help='Boolean flag using type=bool (problematic)')
    
    # Correct way: using action='store_true'
    parser.add_argument('--store-true-flag', action='store_false',
                      help='Boolean flag using action=store_true (correct)')
    
    # Alternative: using action='store_false' with a different default
    parser.add_argument('--no-flag', dest='use_flag', action='store_false', default=True,
                      help='Boolean flag using action=store_false (correct)')

    # Parse the arguments
    args = parser.parse_args()
    
    print("\n=== Boolean flag behavior demonstration ===\n")
    
    # Show the parsed values
    print(f"--type-bool-flag = {args.type_bool_flag} (type: {type(args.type_bool_flag).__name__})")
    print(f"--store-true-flag = {args.store_true_flag} (type: {type(args.store_true_flag).__name__})")
    print(f"--no-flag (use_flag) = {args.use_flag} (type: {type(args.use_flag).__name__})")
    
    # Explain the issue
    if args.type_bool_flag and '--type-bool-flag False' in ' '.join(__import__('sys').argv):
        print("\n⚠️ ISSUE DETECTED: '--type-bool-flag False' was passed but the value is True!")
        print("This is because bool('False') is True in Python, any non-empty string evaluates to True.")
        print("This is why 'type=bool' should never be used in argparse.")
    
    # Show what's happening with bool()
    print("\n=== How bool() evaluates strings in Python ===")
    print(f"bool('False') → {bool('False')}")
    print(f"bool('false') → {bool('false')}")
    print(f"bool('0')     → {bool('0')}")
    print(f"bool('no')    → {bool('no')}")
    print(f"bool('')      → {bool('')}  # Only empty string is False")
    
    print("\n=== Best practices ===")
    print("1. Use action='store_true' for flags that should be False by default")
    print("2. Use action='store_false' for flags that should be True by default")
    print("3. NEVER use type=bool for command line flags")

if __name__ == "__main__":
    main() 