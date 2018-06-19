import requests
import pandas
import json

name = ('cell')
file = open(name, 'r')
content = file.readlines()

outFile = open("cell_location", 'r')
out_content = outFile.readlines()

outFile_final = open("cell_location_final", 'w')

firstCell = True
oldCell = 0
index = 1

def cell_loc (cellid,lac,mnc):
	with open('data.json', 'r') as f:
		data = json.load(f)
		data['homeMobileCountryCode'] = 466
		data['homeMobileNetworkCode'] = mnc
		data['radioType'] = "lte"
		data['carrier'] = "Chunghwa"
		data['cellTowers'][0]['cellId'] = cellid
		data['cellTowers'][0]['locationAreaCode'] = lac
		data['cellTowers'][0]['mobileCountryCode'] = 466
		data['cellTowers'][0]['mobileNetworkCode'] = mnc

	#print (data)

	f= "{\n" +\
	"\"homeMobileCountryCode\": 466,\n" + \
	"\"homeMobileNetworkCode\":"+ str(int(mnc))+",\n" + \
	"\"radioType\": \"lte\",\n" +\
	"  \"cellTowers\": [\n" + \
	"    {\n" + \
	"      \"cellId\": "+ str(int(cellid)) +",\n" + \
	"      \"locationAreaCode\":  "+ str(int(lac)) + ",\n" + \
	"      \"mobileCountryCode\": 466,\n" + \
	"      \"mobileNetworkCode\":" + str(int(mnc))+"\n" + \
	"    }\n" + \
	"  ]\n" + \
	"}"
	
	#print (f)

	#print (data)

	r = requests.post("https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyAv9wpWpYtIoaXJKyVmMEUjwUFy3tK-vLM"
				, json=data)
	response_json= r.json()
	lat,lng,acc = response_json['location']['lat'],response_json['location']['lng'], response_json['accuracy']

	print (str(mnc)+'\t' + str(cellid) + "\t" + str(lac) + "\t" + str(lat)+ "\t" +str(lng)+ "\t" +str(acc))

	return lat,lng,acc


dup = False
for x in range(len(content)):
	line=content[x].split('\t')

	data = line[2].split('\n')[0]+ "\t" + line[0] + "\t" + line[1]  
	lat,lng,acc = cell_loc(line[0], line[1], line[2].split('\n')[0])
	
	for index in range(len(out_content)):
		if out_content[index].split('\t')[1] == line[0]:
			dup = True
			print 'dup!!'
	if dup is False:
		data = data + "\t" + str(lat)+ "\t" +str(lng)+ "\t" +str(acc) +"\n"
		outFile_final.write(data)

file.close()
outFile.close()
outFile_final.close()



url = "https://us1.unwiredlabs.com/v2/process.php"

# payload = "{\"token\": \"92d8d3bdcd3fb3\",\"radio\": \"lte\", \
# 			\"mcc\": 466,\"mnc\": 97,\"cells\": \
# 			[{\"lac\": 11290,\"cid\": 27166952, \"psc\": 475}],\"address\": 1}"


data= {"token": "92d8d3bdcd3fb3", \
		"radio": "lte", \
		"mcc": 466, \
		"mnc": 97, \
		"cells": [{"lac": 11290, \
					"cid": 27166952, \
					"psc": 475
					}], \
		"address": 1 \
		}			
response = requests.request("POST", url, json=data)
response = response.json()
# response = json.dumps(response.text)

# #df = pandas.read_json(response,encoding=encoding)
print(response['lat'])