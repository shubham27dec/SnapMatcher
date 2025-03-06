from imagecluster import calc, io, postproc
from imagecluster.feedback import FeedbackSystem
from imagecluster.learning import LearningSystem
import matplotlib.pyplot as plt

def get_user_feedback(clusters):
    """Get feedback for each cluster"""
    feedback = {}
    cluster_num = 1
    
    print("\nProviding feedback for clusters:")
    for csize, cluster_list in clusters.items():
        for cluster in cluster_list:
            print(f"\nCluster {cluster_num} ({len(cluster)} images):")
            while True:
                response = input("Is this grouping good? (y/n): ").lower()
                if response in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")
            feedback[cluster_num] = response == 'y'
            cluster_num += 1
    return feedback

def main():
    # Initialize systems
    feedback_system = FeedbackSystem()
    learning_system = LearningSystem(feedback_system)

    # Load images
    print("Loading images...")
    images, fingerprints, timestamps = io.get_image_data('./test_images/')

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

     # Get feedback after showing clusters
    plt.show(block=False)  # Show but don't block
    feedback = get_user_feedback(clusters)
    
    # Update learning system
    print("Updating learning system with feedback...")
    feedback_system.store_feedback(feedback)
    learning_system.adjust_weights()
    
    print("Feedback processed and learning updated!")
    plt.show()  # Keep window open

if __name__ == "__main__":
    main()