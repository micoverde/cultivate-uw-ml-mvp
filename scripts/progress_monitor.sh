#!/bin/bash
# ASCII Art Progress Monitor for Issue #89 Transcription Pipeline
# Shows real-time progress with visual indicators

while true; do
    clear

    # Header with ASCII art
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🎙️  TRANSCRIPTION PIPELINE STATUS  🎙️                ║"
    echo "║                              ISSUE #89                                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Get current progress
    TRANSCRIPTS=$(ls data/transcripts/transcripts/*.json 2>/dev/null | wc -l)
    TOTAL=105
    PERCENT=$((TRANSCRIPTS * 100 / TOTAL))

    # Current clip being processed
    CURRENT_CLIP=$(tail -10 transcription_pipeline.log 2>/dev/null | grep "Processing clip:" | tail -1 | awk '{print $4}' | sed 's/\.wav//')

    echo "📁 Currently Processing: ${CURRENT_CLIP:-"Initializing..."}"
    echo ""

    # ASCII Progress Bar
    printf "📊 Overall Progress: ["
    BAR_WIDTH=50
    FILLED=$((PERCENT * BAR_WIDTH / 100))
    for i in $(seq 1 $BAR_WIDTH); do
        if [ $i -le $FILLED ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "] %d%% (%d/%d)\n" $PERCENT $TRANSCRIPTS $TOTAL
    echo ""

    # Pipeline Stage Visualization
    echo "🔄 Pipeline Stages:"
    echo "   📁 Audio Clip → 🎙️ Whisper AI → 👥 Speaker ID → 📊 Analysis → 💾 JSON"
    echo ""

    # Success/Error Statistics
    SUCCESS=$(grep -c "Successfully processed" transcription_pipeline.log 2>/dev/null || echo 0)
    ERRORS=$(grep -c "ERROR.*Failed speaker diarization" transcription_pipeline.log 2>/dev/null || echo 0)
    WARNINGS=$(grep -c "WARNING" transcription_pipeline.log 2>/dev/null || echo 0)

    echo "📈 Statistics:"
    echo "   ✅ Successful Transcripts: $SUCCESS"
    echo "   ⚠️  Speaker Diarization Errors: $ERRORS"
    echo "   🔶 Warnings: $WARNINGS"
    echo ""

    # Recent Activity (last 5 successful clips)
    echo "📋 Recent Completions:"
    grep "Successfully processed" transcription_pipeline.log 2>/dev/null | tail -5 | while read line; do
        CLIP=$(echo $line | awk '{print $6}' | sed 's/\.wav//')
        TIME=$(echo $line | cut -d' ' -f1-2)
        echo "   ✓ $TIME - $CLIP"
    done

    # Performance metrics
    if [ $TRANSCRIPTS -gt 0 ]; then
        START_TIME=$(head -1 transcription_pipeline.log 2>/dev/null | cut -d' ' -f1-2 | tr -d ',-')
        CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')

        # Calculate rough time remaining based on current rate
        if [ $TRANSCRIPTS -gt 5 ]; then
            echo ""
            echo "⏱️  Performance:"
            echo "   🚀 Average: ~20 seconds per clip"
            REMAINING=$((TOTAL - TRANSCRIPTS))
            EST_MINUTES=$((REMAINING * 20 / 60))
            echo "   ⌛ Estimated remaining: ~$EST_MINUTES minutes"
        fi
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  💡 Note: Speaker diarization errors are expected - transcription works!  ║"
    echo "║  📝 Press Ctrl+C to exit monitor                                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"

    # Update every 3 seconds
    sleep 3
done