from imagecluster import calc, io, postproc
from imagecluster.feedback import FeedbackSystem
from imagecluster.learning import LearningSystem

def main():
    # Initialize systems
    feedback_system = FeedbackSystem()
    learning_system = LearningSystem(feedback_system)

    # Load images
    print("Loading images...")
    images, fingerprints = io.get_image_data('./test_images/')

    # Smart clustering
    print("Creating smart clusters...")
    clusters = calc.smart_cluster(
        fingerprints,
        feedback_system=feedback_system,
        learning_system=learning_system
    )

    # Show results
    print("Displaying results...")
    postproc.visualize_smart_clusters(clusters, images, feedback_system)