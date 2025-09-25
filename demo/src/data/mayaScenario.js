/**
 * Maya Puzzle Frustration Scenario
 * Official UW Demo Script Requirement
 *
 * Core scenario for MVP Sprint 1 demo where users type responses
 * to Maya's puzzle frustration and receive AI coaching feedback.
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #108 - PIVOT to User Response Evaluation
 */

export const mayaPuzzleScenario = {
  id: "maya-puzzle-frustration",
  title: "Maya's Puzzle Frustration",
  description: "Practice responding to a child's frustration with appropriate coaching and support strategies.",

  // Scenario Context
  setting: "Free play time in preschool classroom",
  childAge: 4,
  situationDuration: "5 minutes",

  // Visual Context (for future illustration)
  visualDescription: "Educator kneeling at child's eye level, child appears frustrated with puzzle pieces scattered, classroom setting with learning materials visible",

  // Audio Transcript (exact from demo script)
  context: "During free play time, 4-year-old Maya is working on a 12-piece puzzle. She has been trying for 5 minutes and is showing signs of frustration.",

  audioTranscript: `Maya: "This is stupid! I can't do it!" *pushes puzzle pieces away*
Maya: *crosses arms and turns away from the table*
Maya: "I'm never gonna be able to do puzzles!"`,

  // User Response Requirements
  responseRequirements: {
    minimumCharacters: 100,
    prompt: "What would you say to the child?",
    considerations: [
      "How would you respond to Maya's frustration?",
      "What actions would you take?",
      "How would you support her learning and emotional regulation?"
    ]
  },

  // Expected Analysis Categories (5 required categories)
  analysisCategories: [
    {
      id: "emotional_support",
      name: "Emotional Support & Validation",
      description: "Recognition and validation of child's emotions",
      weight: 0.25
    },
    {
      id: "scaffolding_support",
      name: "Scaffolding & Learning Support",
      description: "Breaking down tasks and providing appropriate assistance",
      weight: 0.25
    },
    {
      id: "language_quality",
      name: "Language & Communication Quality",
      description: "Developmentally appropriate language and communication",
      weight: 0.20
    },
    {
      id: "developmental_appropriateness",
      name: "Developmental Appropriateness",
      description: "Age-appropriate expectations and responses",
      weight: 0.15
    },
    {
      id: "overall_effectiveness",
      name: "Overall Effectiveness Score",
      description: "Overall quality of pedagogical response",
      weight: 0.15,
      scale: "0-10"
    }
  ],

  // Evidence-Based Benchmarks (from demo script)
  evidenceBasedMetrics: [
    {
      strategy: "emotion_labeling_validation",
      name: "Emotion labeling and validation",
      effectiveness: 89,
      description: "Explicitly acknowledging and naming child's emotions"
    },
    {
      strategy: "concrete_next_steps",
      name: "Offering concrete next steps",
      effectiveness: 76,
      description: "Providing specific, actionable suggestions"
    },
    {
      strategy: "effort_celebration",
      name: "Celebrating effort over outcome",
      effectiveness: 82,
      description: "Focusing on process rather than results"
    }
  ],

  // Sample High-Quality Response (from demo script)
  exemplarResponse: `Maya, I can see you're feeling really frustrated with that puzzle. That's okay - puzzles can be tricky! I noticed you got these three pieces to fit perfectly. Would you like to try finding one more piece that fits, or would you like to take a break and come back to it later?`,

  // Common Response Patterns for Testing
  testResponsePatterns: {
    highQuality: {
      characteristics: ["emotion acknowledgment", "specific praise", "choices offered", "calm tone"],
      expectedScore: "7.5-9.0"
    },
    mediumQuality: {
      characteristics: ["some validation", "general encouragement", "task redirection"],
      expectedScore: "5.0-7.4"
    },
    lowQuality: {
      characteristics: ["dismissive", "no emotion validation", "pressure to continue"],
      expectedScore: "0-4.9"
    }
  },

  // Pedagogical Learning Objectives
  learningObjectives: [
    "Practice emotion coaching and validation techniques",
    "Apply scaffolding strategies for challenging tasks",
    "Develop age-appropriate communication skills",
    "Learn to offer choices and maintain child autonomy",
    "Understand when to persist vs. when to take breaks"
  ],

  // Related Research Citations
  researchBase: [
    "Emotion coaching effectiveness (Gottman & DeClaire, 1997)",
    "Scaffolding in early childhood (Vygotsky Zone of Proximal Development)",
    "Choice and autonomy in learning (Self-Determination Theory)"
  ]
};

export default mayaPuzzleScenario;