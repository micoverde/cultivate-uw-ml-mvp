/**
 * Realistic Educator Scenarios for Professional Demo
 *
 * Feature 3: Professional Demo Interface
 * Story 3.1: Select realistic educator scenarios
 *
 * Each scenario demonstrates different aspects of ML analysis capabilities
 * across diverse educational contexts and quality levels.
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #48 - Scenario Selection Interface
 */

export const educatorScenarios = [
  // EXEMPLARY TEACHING - Demonstrates high-quality interactions
  {
    id: "exemplary-science-discovery",
    title: "Science Discovery with 4-year-olds",
    description: "Exemplary educator guides children through hands-on exploration of floating and sinking objects.",

    // Educational Context
    ageGroup: "4-5",
    interactionType: "lesson",
    settingType: "small-group",

    // Quality Indicators
    expectedQuality: "exemplary",
    focusAreas: ["questioning", "scaffolding", "emotional-support"],

    // Content
    transcript: `Teacher: I wonder what will happen when we put this wooden block in the water. What do you think, Maria?

Maria: It will go down?

Teacher: That's an interesting prediction! Let's test it out and see what happens. [pauses for 3 seconds]

Child: Look! It's floating!

Teacher: Wow! You noticed something different than what Maria predicted. What do you observe about the block now?

Child: It's staying on top of the water!

Teacher: Exactly! It's floating on top. Now I'm curious - why do you think the wood block floats while this metal spoon sank?

Maria: Maybe because the wood is lighter?

Teacher: That's wonderful thinking, Maria! You're noticing something about the materials. What else do you notice about how they feel different?

Child: The wood feels softer and the spoon feels hard and heavy.

Teacher: You're both making such careful observations! Let's try one more thing and see if we can figure out this floating mystery together.`,

    backgroundContext: "Small group science exploration with 4 children aged 4-5 years. Teacher is facilitating discovery learning about density and floating/sinking properties using everyday objects.",

    participantInfo: {
      educatorExperience: "5+ years early childhood, Masters in Early Childhood Education",
      childCharacteristics: ["Curious and engaged", "Diverse language backgrounds", "Developing scientific reasoning"],
      sessionGoals: ["Develop observation skills", "Practice hypothesis formation", "Build scientific vocabulary"]
    },

    expectedInsights: [
      "High-quality open-ended questioning (85%+)",
      "Excellent wait time and scaffolding techniques",
      "Strong emotional support and encouragement",
      "Effective use of children's ideas to guide learning"
    ],

    duration: 180, // 3 minutes
    complexity: "moderate",
    tags: ["science", "STEM", "inquiry-based", "small-group", "exemplary"]
  },

  // PROFICIENT PRACTICE - Good teaching with room for improvement
  {
    id: "proficient-math-counting",
    title: "Math Counting Activity",
    description: "Proficient educator leads counting practice with manipulatives, showing good interaction quality with some missed opportunities.",

    ageGroup: "3-4",
    interactionType: "lesson",
    settingType: "small-group",

    expectedQuality: "proficient",
    focusAreas: ["questioning", "wait-time"],

    transcript: `Teacher: Today we're going to count these colorful bears! Who can count how many red bears there are?

Child: One, two, three, four!

Teacher: Good job! There are four red bears. Now let's count the blue bears. Ready?

Children: [together] One, two, three!

Teacher: Right, three blue bears. Now, if we put all the red bears and blue bears together, how many do we have?

Child: Seven!

Teacher: Let's count them together to check. One, two, three, four, five, six, seven. Yes! Seven bears altogether.

Child: Can we count the yellow ones now?

Teacher: Sure! How many yellow bears do you see?

Child: Five!

Teacher: Let's count together. One, two, three, four, five. Good counting! Now we know we have red bears, blue bears, and yellow bears.`,

    backgroundContext: "Math center activity with 3-4 year olds focusing on counting and early addition concepts using colorful bear manipulatives.",

    participantInfo: {
      educatorExperience: "3 years early childhood, Bachelor's in Elementary Education",
      childCharacteristics: ["Enthusiastic about math activities", "Mixed counting abilities", "Some need individual support"],
      sessionGoals: ["Practice counting 1-10", "Introduce simple addition", "Build number recognition"]
    },

    expectedInsights: [
      "Moderate open-ended questioning (65%)",
      "Some missed opportunities for deeper thinking",
      "Good use of manipulatives and concrete materials",
      "Could benefit from longer wait times"
    ],

    duration: 150,
    complexity: "simple",
    tags: ["math", "counting", "manipulatives", "small-group", "proficient"]
  },

  // DEVELOPING SKILLS - Shows areas needing improvement
  {
    id: "developing-story-time",
    title: "Story Time Challenges",
    description: "Developing educator reads to children but misses opportunities for interaction and engagement.",

    ageGroup: "5-6",
    interactionType: "reading",
    settingType: "classroom",

    expectedQuality: "developing",
    focusAreas: ["questioning", "wait-time", "emotional-support"],

    transcript: `Teacher: Today we're reading 'The Very Hungry Caterpillar.' Everyone sit criss-cross.

Teacher: [reading] "In the light of the moon a little egg lay on a leaf. One Sunday morning the warm sun came up and - pop! - out of the egg came a tiny and very hungry caterpillar."

Child: I have that book at home!

Teacher: That's nice. [continues reading] "He started to look for some food. On Monday he ate through one apple. But he was still hungry."

Child: Why was he hungry?

Teacher: Because he's a caterpillar. [continues reading] "On Tuesday he ate through two pears, but he was still hungry. On Wednesday he ate through three plums..."

Child: Can we count how many things he ate?

Teacher: Maybe later. Let's finish the story first. [continues reading without pausing for discussion]

Teacher: "And now he wasn't hungry any more - and he wasn't a little caterpillar any more. He was a big, fat caterpillar." The end. Time to go wash hands for snack.`,

    backgroundContext: "Whole group story time with 5-6 year olds during morning circle time. Teacher focuses on reading completion rather than comprehension or interaction.",

    participantInfo: {
      educatorExperience: "1 year early childhood, Bachelor's in Child Development",
      childCharacteristics: ["Eager to participate and share ideas", "Different attention spans", "Strong interest in stories"],
      sessionGoals: ["Develop listening skills", "Build vocabulary", "Encourage love of reading"]
    },

    expectedInsights: [
      "Limited open-ended questioning (25%)",
      "Missed opportunities for children's connections",
      "Minimal wait time for processing",
      "Could enhance emotional support and validation"
    ],

    duration: 120,
    complexity: "simple",
    tags: ["reading", "literacy", "whole-group", "developing", "missed-opportunities"]
  },

  // COMPLEX INTERACTION - Advanced educational scenario
  {
    id: "complex-problem-solving",
    title: "STEM Tower Building Challenge",
    description: "Advanced educator facilitates complex problem-solving with 6-year-olds building towers with limited materials.",

    ageGroup: "6-7",
    interactionType: "problem-solving",
    settingType: "small-group",

    expectedQuality: "proficient",
    focusAreas: ["scaffolding", "questioning", "emotional-support"],

    transcript: `Teacher: Engineers, we have a challenge today! Your job is to build the tallest tower possible using only these 20 marshmallows and 15 toothpicks.

Child A: Can we use all the marshmallows?

Teacher: What do you think? What might happen if you use them all at the bottom?

Child B: It might fall over because it's too heavy!

Teacher: Interesting hypothesis! How could you test that idea?

Child A: We could try building one with lots at the bottom and one with just a few?

Teacher: What excellent scientific thinking! How else might you plan your tower before you start building?

Child C: We could draw it first?

Teacher: That's a wonderful idea. What would drawing help you do?

Child C: See if it might work before we waste marshmallows.

Teacher: I love how you're thinking about planning and testing! [5 minutes later] I notice some towers falling down. What patterns are you noticing?

Child B: The wide bottom ones stay up better.

Child A: And if you push the toothpicks in too far, the marshmallow breaks!

Teacher: You're making such important discoveries! What could you try differently based on what you've learned?`,

    backgroundContext: "STEM center with small group of 6-7 year olds working on engineering design challenge. Emphasis on planning, testing, and iteration.",

    participantInfo: {
      educatorExperience: "8+ years, STEM specialist certification",
      childCharacteristics: ["Strong problem-solving interest", "Collaborative workers", "Different engineering experience levels"],
      sessionGoals: ["Practice engineering design process", "Develop collaboration skills", "Build persistence through iteration"]
    },

    expectedInsights: [
      "Sophisticated scaffolding techniques (80%+)",
      "Excellent use of children's ideas for learning",
      "Strong emotional support for risk-taking",
      "Advanced questioning that promotes deeper thinking"
    ],

    duration: 300, // 5 minutes
    complexity: "complex",
    tags: ["STEM", "engineering", "problem-solving", "collaboration", "advanced"]
  },

  // TRANSITION MANAGEMENT - Classroom management scenario
  {
    id: "transition-cleanup",
    title: "Clean-up Transition",
    description: "Educator manages transition from free play to circle time, showing positive behavior guidance strategies.",

    ageGroup: "4-5",
    interactionType: "transition",
    settingType: "classroom",

    expectedQuality: "proficient",
    focusAreas: ["emotional-support", "classroom-organization"],

    transcript: `Teacher: Friends, in two minutes it will be time to clean up for circle time. What do you need to finish up?

Child A: I'm still building my castle!

Teacher: I can see you're working hard on that castle! You have two minutes to find a good stopping point. What could you do with your castle so it's safe?

Child A: Put it on the high shelf?

Teacher: That sounds like a good plan! [singing] "Clean-up time, clean-up time, everybody do your share..."

Child B: [upset] I don't want to stop painting!

Teacher: You're disappointed about stopping your painting, aren't you? I can see you were really enjoying that. How can we take care of your painting so you can finish it later?

Child B: Put it in my cubby?

Teacher: Exactly! Your painting will be safe in your cubby. Sarah, I notice you're helping put the blocks away even though you weren't playing with them. That's very thoughtful!

Child C: Where do these puzzles go?

Teacher: Where do you think they belong? Look around the room for clues.

Child C: On the puzzle shelf!

Teacher: Perfect! You remembered where they belong.`,

    backgroundContext: "Transition time from free play centers to whole group circle time. Teacher uses positive behavior guidance and problem-solving support.",

    participantInfo: {
      educatorExperience: "4 years early childhood, positive behavior support training",
      childCharacteristics: ["Different transition needs", "Some difficulty with changes", "Generally cooperative group"],
      sessionGoals: ["Develop independence in cleanup", "Practice emotional regulation", "Build classroom community"]
    },

    expectedInsights: [
      "Strong emotional validation and support (75%)",
      "Good use of choices and problem-solving",
      "Effective positive reinforcement strategies",
      "Clear expectations with flexibility"
    ],

    duration: 180,
    complexity: "moderate",
    tags: ["transitions", "behavior-guidance", "emotional-support", "independence"]
  },

  // OUTDOOR LEARNING - Nature-based exploration
  {
    id: "outdoor-nature-exploration",
    title: "Garden Investigation",
    description: "Educator leads outdoor learning experience with 3-year-olds exploring insects and plants in the school garden.",

    ageGroup: "3-4",
    interactionType: "play",
    settingType: "outdoor",

    expectedQuality: "exemplary",
    focusAreas: ["questioning", "scaffolding", "emotional-support"],

    transcript: `Teacher: Look what I found under this log! What do you notice?

Child: Bugs! Lots of bugs!

Teacher: Yes, there are several small creatures here. What do you observe about how they move?

Child: They're crawling really fast!

Teacher: They are moving quickly! I wonder why they're moving so fast now?

Child: Maybe they're scared of us?

Teacher: That's thoughtful thinking! What could we do to help them feel safer?

Child: Be really quiet and still?

Teacher: Let's try that. [whispers] Let's watch very quietly and see what happens.

[pause]

Child: [whispering] One is going under the dirt!

Teacher: You're such a careful observer! What do you think it's doing under there?

Child: Maybe it lives there?

Teacher: That's a wonderful idea! Many creatures do make their homes in the soil. What else do you think might live in the ground here in our garden?

Child: Worms?

Teacher: Should we look for some worms to observe too?`,

    backgroundContext: "Outdoor learning in school garden area with small group of 3-4 year olds. Focus on natural science exploration and observation skills.",

    participantInfo: {
      educatorExperience: "6 years, nature-based early learning specialization",
      childCharacteristics: ["High curiosity about nature", "Some initial hesitation with insects", "Strong sensory learners"],
      sessionGoals: ["Develop observation skills", "Build comfort with nature", "Practice gentle investigation"]
    },

    expectedInsights: [
      "Excellent questioning that follows children's interests (90%)",
      "Sensitive scaffolding for nature exploration",
      "Strong emotional support for different comfort levels",
      "Effective use of outdoor environment for learning"
    ],

    duration: 240,
    complexity: "moderate",
    tags: ["outdoor", "nature", "science", "observation", "exemplary"]
  },

  // ONE-ON-ONE SUPPORT - Individual scaffolding scenario
  {
    id: "individual-writing-support",
    title: "Writing Conference Support",
    description: "Educator provides individual support to a 5-year-old working on early writing skills during writing workshop.",

    ageGroup: "5-6",
    interactionType: "lesson",
    settingType: "one-on-one",

    expectedQuality: "proficient",
    focusAreas: ["scaffolding", "emotional-support"],

    transcript: `Teacher: I see you're working on your story about your dog, Marcus. Tell me about what you've written so far.

Marcus: I wrote "My dog is big" but I can't write "fluffy."

Teacher: You wrote "My dog is big" - that tells me something important about your dog! What sounds do you hear at the beginning of "fluffy"?

Marcus: "Fff"?

Teacher: Yes! What letter makes that "fff" sound?

Marcus: F!

Teacher: Exactly! Write that F down. Now what sound comes next in "fluffy"?

Marcus: "Lll"?

Teacher: Good listening! What letter makes that sound?

Marcus: L?

Teacher: Perfect! You're doing such careful sound work. Write that L next to your F.

Marcus: F-L... now what?

Teacher: Keep listening to the sounds. What do you hear next?

Marcus: "Uh"?

Teacher: Yes! That middle sound is tricky. The letter U can make that sound. But you know what? Even if you can't hear every sound perfectly, your reader will understand "FL" for fluffy. That's excellent beginning writing!

Marcus: So I can just write F-L?

Teacher: Absolutely! That shows you understand how letters and sounds work together.`,

    backgroundContext: "Individual writing conference during writing workshop time. Teacher supports emergent writing development through phonemic awareness and encouragement.",

    participantInfo: {
      educatorExperience: "7 years, reading specialist background",
      childCharacteristics: ["Emerging writer, strong verbal skills", "Perfectionist tendencies", "Enjoys storytelling"],
      sessionGoals: ["Build phonemic awareness", "Encourage risk-taking in writing", "Develop writing confidence"]
    },

    expectedInsights: [
      "Excellent individualized scaffolding (85%)",
      "Strong emotional support for learning risks",
      "Effective phonics instruction embedded in authentic writing",
      "Good balance of challenge and support"
    ],

    duration: 150,
    complexity: "moderate",
    tags: ["writing", "individual", "scaffolding", "literacy", "phonics"]
  },

  // SOCIAL-EMOTIONAL LEARNING - Conflict resolution
  {
    id: "conflict-resolution",
    title: "Playground Conflict Resolution",
    description: "Educator helps two 4-year-olds resolve a conflict over playground equipment using problem-solving strategies.",

    ageGroup: "4-5",
    interactionType: "problem-solving",
    settingType: "outdoor",

    expectedQuality: "exemplary",
    focusAreas: ["emotional-support", "scaffolding"],

    transcript: `Teacher: I see two upset children here. Emma, you look frustrated, and Jake, you look angry too. What's happening?

Emma: Jake won't let me have a turn on the swing!

Jake: I was here first! She can wait!

Teacher: So Emma, you want a turn on the swing, and Jake, you feel like you should keep swinging because you were first. Both of those feelings make sense.

Emma: But I've been waiting forever!

Teacher: Emma, it sounds like waiting is really hard for you right now. Jake, how long have you been swinging?

Jake: Since... um... since snack time?

Teacher: That has been a while. Jake, what do you think might be a fair solution for both of you?

Jake: Maybe... she can have five more pushes after I'm done?

Teacher: That's one idea. Emma, how does that sound to you?

Emma: But when will you be done?

Teacher: That's a good question, Emma. Jake, how will Emma know when your turn is finished?

Jake: I could count to 50 more pushes?

Emma: That's still a lot!

Teacher: You both care about being fair. What would feel fair to both of you?

Emma: Maybe 25?

Jake: Okay, 25 more, then Emma gets a long turn.

Teacher: You two solved that problem together! How are you feeling now?`,

    backgroundContext: "Outdoor playground conflict resolution between two 4-year-olds during free play time. Teacher facilitates problem-solving and emotional regulation.",

    participantInfo: {
      educatorExperience: "5+ years, social-emotional learning specialization",
      childCharacteristics: ["Strong-willed children", "Developing social skills", "Generally good problem-solvers"],
      sessionGoals: ["Practice conflict resolution", "Develop empathy", "Build problem-solving skills"]
    },

    expectedInsights: [
      "Excellent emotional validation for both children (95%)",
      "Sophisticated scaffolding of problem-solving process",
      "Strong facilitation without imposing adult solutions",
      "Effective social-emotional learning integration"
    ],

    duration: 200,
    complexity: "complex",
    tags: ["social-emotional", "conflict-resolution", "problem-solving", "outdoor", "exemplary"]
  }
];

// Helper functions for filtering and categorization
export const getScenariosByAge = (ageGroup) => {
  return educatorScenarios.filter(scenario => scenario.ageGroup === ageGroup);
};

export const getScenariosByQuality = (quality) => {
  return educatorScenarios.filter(scenario => scenario.expectedQuality === quality);
};

export const getScenariosByType = (interactionType) => {
  return educatorScenarios.filter(scenario => scenario.interactionType === interactionType);
};

export const getQualityColor = (quality) => {
  const colors = {
    exemplary: '#10B981', // Green
    proficient: '#3B82F6', // Blue
    developing: '#F59E0B', // Amber
    struggling: '#EF4444'  // Red
  };
  return colors[quality] || '#6B7280';
};

export const getAgeGroupLabel = (ageGroup) => {
  const labels = {
    '2-3': 'Ages 2-3 years',
    '3-4': 'Ages 3-4 years',
    '4-5': 'Ages 4-5 years',
    '5-6': 'Ages 5-6 years',
    '6-7': 'Ages 6-7 years',
    '7-8': 'Ages 7-8 years'
  };
  return labels[ageGroup] || ageGroup;
};

export const getInteractionTypeLabel = (type) => {
  const labels = {
    lesson: 'Structured Lesson',
    play: 'Free Play',
    reading: 'Reading Time',
    'problem-solving': 'Problem Solving',
    transition: 'Transitions',
    'one-on-one': 'Individual Support'
  };
  return labels[type] || type;
};