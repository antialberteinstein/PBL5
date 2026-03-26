import numpy as np

# Standard Pose Categories
POSES = ["Nhin thang", "Ngang len", "Cui xuong", "Quay trai", "Quay phai"]

def get_pose_name(pose: np.ndarray) -> str:
    """
    Classifies a [pitch, yaw, roll] array into 5 main categories.
    
    Args:
        pose: Array containing [pitch, yaw, roll]
        
    Returns:
        String name of the pose category
    """
    if pose is None:
        return "Nhin thang"
    
    pitch, yaw, roll = pose[0], pose[1], pose[2]
    
    # InsightFace's pose outputs are (pitch, yaw, roll). 
    # Positive pitch = looking up (ngang len), Negative pitch = looking down (cui xuong)
    if pitch > 15:
        return "Ngang len"
    elif pitch < -15:
        return "Cui xuong"
    elif yaw > 20:   # Positive yaw = looking left (quay trai)
        return "Quay trai"
    elif yaw < -20:  # Negative yaw = looking right (quay phai)
        return "Quay phai"
    else:
        return "Nhin thang"
