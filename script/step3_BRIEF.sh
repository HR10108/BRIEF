#!/usr/bin/env bash
CONFIG_FILE="${1:-./default_config.sh}"
source "$CONFIG_FILE"

echo "Beginning BRIEF coreset selection for model: ${MODEL_BRIEF}"
BRIEF_JOB_NAME="${MODEL_BRIEF}-BRIEF"
LOG_FILE=$(log_file "$BRIEF_JOB_NAME")

# Create output directory
mkdir -p "$SAVE_DIR"

# Build arguments for run_BRIEF.py
BRIEF_ARGS=(
    "--grad_dir" "$GRADIENT_PATH"
    "--proj_dim" "$DIM"
    "--device" "$DEVICE"
    "--batch_size" "$BATCH_SIZE"
    "--proportions" $PROPORTIONS
    "--output_dir" "$SAVE_DIR"
    "--grad_type" "$GRAD_TYPE"
    "--seed" "$SEED"
)

# Add optional arguments if enabled
if [ "$PREFIX" != "" ]; then
    BRIEF_ARGS+=("--prefix" "$PREFIX")
fi

if [ "$ENABLE_PLOT" = "true" ]; then
    BRIEF_ARGS+=("--plot")
fi

if [ "$ENABLE_AUTO_SEARCH" = "true" ]; then
    BRIEF_ARGS+=("--auto_search")
    BRIEF_ARGS+=("--search_precision" "$SEARCH_PRECISION")
    BRIEF_ARGS+=("--max_iterations" "$MAX_ITERATIONS")
fi

echo "Running BRIEF with arguments: ${BRIEF_ARGS[@]}"

# Execute BRIEF
python "$BRIEF_PYTHON_PATH" "${BRIEF_ARGS[@]}" > "$LOG_FILE" 2>&1

# Check if BRIEF completed successfully
if [ $? -eq 0 ]; then
    echo "BRIEF coreset selection completed successfully!"
    echo "Output saved to: $SAVE_DIR"
    echo "Log file: $LOG_FILE"
    
    # List generated files
    echo "Generated files:"
    ls -la "$SAVE_DIR"/*
    
    # Determine the coreset filename based on the mode and save it for step4
    CORESET_CONFIG_FILE="${BASE_DIR}/coreset_path.conf"
    
    if [ "$ENABLE_AUTO_SEARCH" = "true" ]; then
        # Auto-search mode: find the auto_alpha file for the specified proportion
        PROPORTION_INT=$(echo "$PROPORTIONS" | awk '{print int($1*100)}')
        if [ "$PREFIX" != "" ]; then
            CORESET_PATTERN="${SAVE_DIR}/${PREFIX}_auto_alpha*_p${PROPORTION_INT}_idx.npz"
        else
            CORESET_PATTERN="${SAVE_DIR}/auto_alpha*_p${PROPORTION_INT}_idx.npz"
        fi
        
        # Find the actual generated file
        CORESET_FILE=$(ls -1 $CORESET_PATTERN 2>/dev/null | head -n1)
        
        if [ -n "$CORESET_FILE" ]; then
            echo "Found auto-search coreset file: $CORESET_FILE"
            echo "CORESET_FILE=\"$CORESET_FILE\"" > "$CORESET_CONFIG_FILE"
            echo "Coreset path saved to: $CORESET_CONFIG_FILE"
        else
            echo "Warning: Could not find auto-search coreset file matching pattern: $CORESET_PATTERN"
            echo "Available files:"
            ls -la "$SAVE_DIR"/*.npz 2>/dev/null || echo "No .npz files found"
        fi
    else
        # Regular mode: construct the filename based on alpha and proportion
        ALPHA_INT=$(echo "$ALPHA" | awk '{print int($1*1000)}')
        PROPORTION_INT=$(echo "$PROPORTIONS" | awk '{print int($1*100)}')
        if [ "$PREFIX" != "" ]; then
            CORESET_FILE="${SAVE_DIR}/${PREFIX}_alpha${ALPHA_INT}_p${PROPORTION_INT}_idx.npz"
        else
            CORESET_FILE="${SAVE_DIR}/alpha${ALPHA_INT}_p${PROPORTION_INT}_idx.npz"
        fi
        
        if [ -f "$CORESET_FILE" ]; then
            echo "Found regular coreset file: $CORESET_FILE"
            echo "CORESET_FILE=\"$CORESET_FILE\"" > "$CORESET_CONFIG_FILE"
            echo "Coreset path saved to: $CORESET_CONFIG_FILE"
        else
            echo "Warning: Expected coreset file not found: $CORESET_FILE"
            echo "Available files:"
            ls -la "$SAVE_DIR"/*.npz 2>/dev/null || echo "No .npz files found"
        fi
    fi
    
else
    echo "BRIEF coreset selection failed. Check log file: $LOG_FILE"
    exit 1
fi
