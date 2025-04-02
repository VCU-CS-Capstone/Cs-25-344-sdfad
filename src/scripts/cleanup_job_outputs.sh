#!/bin/bash

# Usage: ./cleanup_job_outputs.sh <job_name>

if [ -z "$1" ]; then
  echo "❌ Usage: $0 <job_name>"
  exit 1
fi

JOB_NAME="$1"

echo "🧹 Cleaning up .out, .err, and .log files for job: $JOB_NAME"

# Target patterns in the current directory
rm -v "${JOB_NAME}"_*.out 2>/dev/null
rm -v "${JOB_NAME}"_*.err 2>/dev/null
rm -v "${JOB_NAME}"_*.log 2>/dev/null

echo "✅ Cleanup complete."
