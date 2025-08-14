def lm_xy(landmark, width, height): # pose result -> (x,y) pixsel
    return int(landmark.x * width), int(landmark.y * height)


