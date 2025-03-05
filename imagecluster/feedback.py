import time
import pickle

class FeedbackSystem:
    def __init__(self):
        self.feedback_history = []
        self.group_weights = {
            'similarity': 0.5,
            'pattern': 0.5,
            'structure': 0.5
        }
    
    def store_feedback(self, group_id, is_good):
        feedback = {
            'group_id': group_id,
            'accepted': is_good,
            'timestamp': time.time()
        }
        self.feedback_history.append(feedback)
        
    def get_cluster_confidence(self, cluster_id):
        # Calculate confidence based on past feedback
        relevant_feedback = [f for f in self.feedback_history 
                           if f['group_id'] == cluster_id]
        if not relevant_feedback:
            return 0.5
        
        success_rate = sum(f['accepted'] for f in relevant_feedback) / len(relevant_feedback)
        return success_rate