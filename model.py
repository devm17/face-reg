import pickle
import face_recognition
import glob
import numpy as np

list_of_files = [i for i in glob.glob('*.jpg')]
faces_encodings = []
names = list_of_files.copy()

for i in range(len(list_of_files)):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])

unknown_image = face_recognition.load_image_file("test_data/test.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces(faces_encodings, unknown_face_encoding)
    
flag = False
for i in results:
    if i==True:
        flag = True
        face_distances = face_recognition.face_distance(faces_encodings, unknown_face_encoding)
        index = np.argmin(face_distances)
        if results[index]:
            print(names[index])
        break
        
if flag == False:
    print("It's a unique face")

with open ("face_reg","wb") as f:
    pickle.dump(faces_encodings,f)