import numpy as np
import cv2

colors = {"BLUE":(255,0,0), "GREEN":(0,255,0), "RED":(0,0,255), "YELLOW":(0,255,255), "PURPLE":(240, 0, 159)}


""" Process output for models detecting bounding boxes. """
def process_output_bb(output, count, tracker, frame_copy, height, width, args, image_flag, ft, persons, PDT):
    
    f = "frame"+str(count) 
    if count == 1:
        last_f = f
        ft = 0
    else:
        last_f = "frame"+str(count-1)
        
    tracker[f] = {}

    cv2.putText(frame_copy, "Frame{}   Time:{:.3f}sec".format(int(count), ft/1000), (width//100, height//10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["GREEN"], 2)   

    person_id = 0 
    
    if image_flag:
        print("output shape:", output.shape)
    # Update the frame to include detected bounding boxes and tracker
    for i in range(len(output[0][0])):    # Output shape: 1x1xNofBoxesx7                   
        confidence = output[0][0][i][2]

        if confidence >= args.pt:
                     
            xmin = int(output[0][0][i][3] * width)
            ymin = int(output[0][0][i][4] * height)
            xmax = int(output[0][0][i][5] * width)
            ymax = int(output[0][0][i][6] * height)
            xc, yc = int((xmax-xmin)/2) + xmin, int((ymax-ymin)/2) + ymin
            
            # updates person_id and localization of a bounding box central point (xc,yc) at the frame
            if persons : # if list of persons is not empty
                j = 0
                not_found = True
                while not_found:
                    p = persons[j]
                    if (abs(p[1]-xc) < 50) and (abs(p[2]-yc) < 50):
                        person_id = p[0]
                        p[1], p[2] = xc, yc
                        not_found = False
                        continue
                    elif j < len(persons)-1:
                        j += 1
                    else:
                        persons.append([len(persons), xc , yc])
                        person_id = persons[-1][0]
                        not_found = False
            else:
                persons.append([0, xc, yc])
                #print(persons)
                person_id = 0
            
            #gets enter time, the time when bb/person entering this f frame
            entering_time = ft 

            # updates the tracker dictionary with people entering_time list         
            person = "person"+str(person_id)
            if person in tracker[last_f]:
                tracker[f][person] = tracker[last_f][person] + [entering_time]
            # adjusts accounting of time for 1-2 frames
            # in case if the model fails to see a person already counted 
            elif count > 2 and person in tracker["frame"+str(count-2)]:
                tracker[f][person] = tracker["frame"+str(count-2)][person] + [entering_time]
            elif count > 3 and person in tracker["frame"+str(count-3)]:
                tracker[f][person] = tracker["frame"+str(count-3)][person] + [entering_time]                
            else:                        
                tracker[f].update({person: [entering_time]})
                
            person_duration_time = tracker[f][person][-1] - tracker[f][person][0]

            """ drawing bounding box and writing frame nr, time, person id, 
                    person duration time,number of people in the frame """
            cv2.rectangle(frame_copy, (xmin,ymin),(xmax,ymax), colors[args.c], 2 )

            cv2.putText(frame_copy, "b{}   t: {:.3f}sec".format(person_id, person_duration_time/1000), (xmin, ymax//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[args.c], 2)
    
    # includes pdt(every person duration time) to the PDT dictionary
    pdt = [(p, (t[-1] - t[0])) for p, t in tracker[f].items()]
    for e in pdt:
        PDT[e[0]] = e[1]
    # computes and writes average duration time for persons who have been detected
    if len(PDT) > 0:
        av = sum([v for v in PDT.values()])/len(PDT)   
        cv2.putText(frame_copy, "b_b model>> Average Duration Time: {:.3f}sec".format(av/1000), (width//100, height//7+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[args.c], 2) 
    cv2.putText(frame_copy, "b_b model>> People in frame: {}".format(len(tracker[f])), (width//100, height//7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[args.c], 2)
    cv2.putText(frame_copy, "b_b model>> Total counted: {}".format(len(persons)), (width//100, height//7+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[args.c], 2)
                      
    return frame_copy, count, tracker, persons, PDT



""" Process output for models detecting human pose. """
def getPoints_toTrackPose(xlist, ylist):
    #makes list of (x,y) points
    l = [[(x,y)] for x,y in zip(xlist,ylist)]

    # sets dict of persons neck coordinates
    points = {} 

    for i in range(1,len(l)):
        # sets first point
        p = 0
        points['p0'] = l[0]
        #if distance between two points <=60 it should be the same person
        not_found = True
        n = 0
        while not_found:   # checks points dict if any neck/point in there has almost the same coordinates as new x,y        
            if (abs(l[i][0][0] - points[list(points)[n]][-1][0]) <= 60) and (abs(l[i][0][1] - points[list(points)[n]][-1][1]) <= 60):
                points[list(points)[n]] += l[i]     #if yes, add new point to person in points dict
                not_found = False
                continue                # ends the loop
            elif n < len(list(points)) - 1:
                n += 1
            else:
                p += 1
                points['p'+str(p)] = l[i]           # if not add new person with new neck point
                not_found = False  
                
    # gets dict "points" with only [xc,yc] coordinates for each person at frame
    c_points = {}
    for p in points:
        c_points[p] = [sum([e[0] for e in points[p]])//len(points[p]), sum([e[1] for e in points[p]])//len(points[p])]
 
    return c_points


def load_poseTracker(count, tracker, frame_copy, height, width, ft, persons, PDT, c_points):
    f = "frame"+str(count) 
    if count == 1:
        last_f = f
    else:
        last_f = "frame"+str(count-1)    
    tracker[f] = {} 
    
    for cp in c_points:
        xc, yc = c_points[cp][0], c_points[cp][1]
        # updates person_id and localization of a neck point (xc,yc) at the frame 
        if persons : # if list of persons is not empty
            j = 0
            not_found = True
            while not_found:
                p = persons[j]
                if (abs(p[1]-xc) < 50) and (abs(p[2]-yc) < 50):
                    person_id = p[0]
                    p[1], p[2] = xc, yc
                    not_found = False
                    continue
                elif j < len(persons)-1:
                    j += 1
                else:
                    persons.append([len(persons), xc , yc])
                    person_id = persons[-1][0]
                    not_found = False
        else:
            persons.append([0, xc, yc])
            person_id = 0

        #gets enter time, the time when bb/person entering this f frame
        entering_time = ft     
        
        # updates the tracker dictionary with people entering_time list         
        person = "person"+str(person_id)
        if person in tracker[last_f]:
            tracker[f][person] = tracker[last_f][person] + [entering_time]
        # adjusts accounting of time for 1-2 frames
        #     in case if the model fails to see a person already counted 
        elif count > 2 and person in tracker["frame"+str(count-2)]:
            tracker[f][person] = tracker["frame"+str(count-2)][person] + [entering_time]
        elif count > 3 and person in tracker["frame"+str(count-3)]:
            tracker[f][person] = tracker["frame"+str(count-3)][person] + [entering_time]                
        else:                        
            tracker[f].update({person: [entering_time]})
            
        person_duration_time = tracker[f][person][-1] - tracker[f][person][0]

        """ drawing and writing frame nr, time, person id, 
                person duration time, number of people in the frame """

        cv2.putText(frame_copy, "p{}   t: {:.3f}sec".format(person_id, person_duration_time/1000), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)            

    # includes pdt(person duration time) to the PDT(People Duration Time) dictionary
    pdt = [(p, (t[-1] - t[0])) for p, t in tracker[f].items()]
    for e in pdt:
        PDT[e[0]] = e[1]
        
    # computes and writes average duration time for all persons who have been detected
    duration = 0
    if len(PDT) > 0:
        duration = (sum([v for v in PDT.values()])/len(PDT)) / 1000
        cv2.putText(frame_copy, "pose model>> Average Duration Time: {:.3f}sec".format(duration), (width//100, height//4+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["YELLOW"], 2)
        
    # computes and writes current_count
    current_count = len(tracker[f])
    cv2.putText(frame_copy, "pose model>> People in frame: {}".format(current_count), (width//100, height//4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["YELLOW"], 2)
    
    # computes and writes total_count
    total_count = len(persons)
    cv2.putText(frame_copy, "pose model>> Total counted: {}".format(total_count), (width//100, height//4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["YELLOW"], 2)        
    
    return count, tracker, persons, PDT, duration, current_count, total_count


def process_pose(output, count, tracker, frame_copy, height, width, ft, persons, PDT):
    
    # Makes heatmap and resize it to the size of the input
    heatmap = np.zeros([output.shape[1], height, width])    
    for h in range(len(output[0])):
        heatmap[h] = cv2.resize(output[0][h], (width, height))

    # Remove final part of output not used for heatmaps
    heatmap = heatmap[:-1]
    
    # Get only pose detections above 0.5 confidence and set it to 255, others to 0
    for e in range(len(heatmap)):
        heatmap[e] = np.where(heatmap[e]>0.5, 255, 0)

    # computes x,y coordinates of people neck points (heatmap[1])
    jointx = []
    jointy = []
    for i in range(len(heatmap[1])):
        for j in range(len(heatmap[1][0])):
            if heatmap[1][i][j] == 255:
                jointx.append(j)
                jointy.append(i)
                
    # gets central joint/neck point for each person at a frame
    c_points = getPoints_toTrackPose(jointx, jointy)
    
    # load tracker
    count, tracker, persons, PDT, duration, current_count, total_count = load_poseTracker(count, tracker, frame_copy, height, width, ft, persons, PDT, c_points)
    
    return frame_copy, count, tracker, persons, PDT, duration, current_count, total_count
