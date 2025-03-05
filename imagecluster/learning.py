class LearningSystem:
    def __init__(self, feedback_system):
        self.feedback = feedback_system
        self.min_weight = 0.1
        self.max_weight = 0.9
        self.learning_rate = 0.1
        
    def adjust_weights(self):
        # Learn from feedback history
        for feedback in self.feedback.feedback_history:
            if feedback['accepted']:
                self.strengthen_patterns(feedback)
            else:
                self.weaken_patterns(feedback)
                
    def strengthen_patterns(self, feedback):
        weights = self.feedback.group_weights
        for key in weights:
            weights[key] = min(
                self.max_weight,
                weights[key] + self.learning_rate
            )
            
    def weaken_patterns(self, feedback):
        weights = self.feedback.group_weights
        for key in weights:
            weights[key] = max(
                self.min_weight,
                weights[key] - self.learning_rate
            )
            
    def get_current_weights(self):
        return self.feedback.group_weights