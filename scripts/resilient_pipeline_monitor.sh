#!/bin/bash
# Resilient Pipeline Monitor with Auto-Recovery
# Monitors transcription pipeline and restarts if stalled

PIPELINE_SCRIPT="src/data_processing/transcription_pipeline.py"
LOG_FILE="transcription_pipeline.log"
MONITOR_LOG="pipeline_monitor.log"
STALL_THRESHOLD=300  # 5 minutes without progress = stalled
CHECK_INTERVAL=30    # Check every 30 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $MONITOR_LOG
}

get_last_activity_time() {
    # Get timestamp of last meaningful activity in log
    local last_entry=$(grep -E "(Transcribed|Successfully processed|Processing clip)" $LOG_FILE | tail -1 | head -c 19)
    if [ -n "$last_entry" ]; then
        date -d "$last_entry" +%s 2>/dev/null || echo 0
    else
        echo 0
    fi
}

get_completed_count() {
    # Count completed transcripts
    ls data/transcripts/transcripts/*.json 2>/dev/null | wc -l
}

is_pipeline_running() {
    # Check if pipeline process is running
    pgrep -f "$PIPELINE_SCRIPT" >/dev/null
}

get_current_clip() {
    # Extract current clip being processed
    grep "Processing clip:" $LOG_FILE | tail -1 | awk '{print $4}' | sed 's/\.wav//' 2>/dev/null
}

restart_pipeline() {
    local reason="$1"
    log_message "🔄 RESTARTING PIPELINE: $reason"

    # Kill existing pipeline
    pkill -f "$PIPELINE_SCRIPT"
    sleep 5

    # Start new pipeline in background
    cd ~/src/tmp2/cultivate-uw-ml-mvp
    source venv/bin/activate
    PYTHONPATH=/home/warrenjo/src/tmp2/cultivate-uw-ml-mvp nohup python $PIPELINE_SCRIPT >> $LOG_FILE 2>&1 &

    log_message "✅ Pipeline restarted with PID: $!"
}

show_status() {
    clear
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    RESILIENT PIPELINE MONITOR                            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Current status
    local completed=$(get_completed_count)
    local current_clip=$(get_current_clip)
    local is_running=$(is_pipeline_running && echo "✅ RUNNING" || echo "❌ STOPPED")
    local last_activity=$(get_last_activity_time)
    local current_time=$(date +%s)
    local time_since_activity=$((current_time - last_activity))

    echo "📊 Pipeline Status: $is_running"
    echo "📁 Completed Transcripts: $completed/105"
    echo "🔄 Current Clip: ${current_clip:-"Unknown"}"
    echo "⏱️ Last Activity: $time_since_activity seconds ago"

    # Progress bar
    local percent=$((completed * 100 / 105))
    printf "\n📈 Progress: ["
    for i in $(seq 1 50); do
        if [ $((percent * 50 / 100)) -ge $i ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "] %d%% (%d/105)\n" $percent $completed

    # Health status
    echo ""
    if [ $time_since_activity -lt $STALL_THRESHOLD ]; then
        echo -e "${GREEN}🟢 Status: HEALTHY${NC}"
    else
        echo -e "${YELLOW}🟡 Status: STALLED (${time_since_activity}s without activity)${NC}"
    fi

    # Recent activity
    echo ""
    echo "📋 Recent Activity:"
    grep -E "(Successfully processed|ERROR)" $LOG_FILE | tail -3 | while read line; do
        if [[ $line == *"ERROR"* ]]; then
            echo -e "  ${RED}❌ $line${NC}"
        else
            echo -e "  ${GREEN}✓ $line${NC}"
        fi
    done
}

# Main monitoring loop
log_message "🚀 Starting resilient pipeline monitor"
log_message "📋 Configuration: Stall threshold=$STALL_THRESHOLD seconds, Check interval=$CHECK_INTERVAL seconds"

# Start pipeline if not running
if ! is_pipeline_running; then
    restart_pipeline "Initial startup"
fi

# Monitoring loop
while true; do
    show_status

    current_time=$(date +%s)
    last_activity=$(get_last_activity_time)
    time_since_activity=$((current_time - last_activity))

    # Check for stalls
    if [ $time_since_activity -gt $STALL_THRESHOLD ]; then
        if is_pipeline_running; then
            restart_pipeline "Stalled for ${time_since_activity} seconds"
            sleep 60  # Give pipeline time to start up
        else
            restart_pipeline "Process died"
        fi
    fi

    # Check if pipeline completed
    completed=$(get_completed_count)
    if [ $completed -eq 105 ]; then
        log_message "🎉 PIPELINE COMPLETED! All 105 transcripts processed"
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                        🎉 PIPELINE COMPLETE! 🎉                         ║${NC}"
        echo -e "${GREEN}║                     All 105 transcripts processed                        ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
        break
    fi

    # Show monitoring footer
    echo ""
    echo "🔄 Monitoring... (Ctrl+C to stop, pipeline continues running)"
    echo "⏰ Next check in $CHECK_INTERVAL seconds"

    sleep $CHECK_INTERVAL
done