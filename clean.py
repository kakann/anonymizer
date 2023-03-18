
import os
import sys

os.getcwd()
duplicate = 0
collection = "/home/martin/Desktop/anonymizer/sweden/images"
for i, filename in enumerate(os.listdir(collection)):
    orgfile= filename
    filename = filename.split("-")[1]
        
        
    if filename.__contains__("("):
        print(filename)
        duplicate += 1
        os.remove("/home/martin/Desktop/anonymizer/sweden/images/" + orgfile)
    else:
        os.rename("/home/martin/Desktop/anonymizer/sweden/images/" + orgfile, "/home/martin/Desktop/anonymizer/sweden/images/" + filename)

    
print(f"Found {duplicate} in the images")
duplicate = 0
collection = "/home/martin/Desktop/anonymizer/sweden/Annotations"
for i, filename in enumerate(os.listdir(collection)):
    orgfile= filename
    filename = filename.split("-")[1]
    os.rename("/home/martin/Desktop/anonymizer/sweden/Annotations/" + orgfile, "/home/martin/Desktop/anonymizer/sweden/Annotations/" + filename)
        
    if filename.__contains__("("):
        print(filename)
        duplicate += 1


print(f"Found {duplicate} in the Annotations")
