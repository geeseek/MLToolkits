from urllib2 import urlopen
import simplejson as json 

key = 'MDExNzEwOTA4MDEzNzMxMTQxNThmNzU5MQ001'

def read(url):
	response = urlopen(url)
	json_obj = json.load(response)
	return json_obj

def write(ouputfile, json_obj):
	f = open(outputfile, 'w')
	f.write(json.dumps(json_obj, indent=4))
	f.close()

def write2txt(outputfile, str):
	f = open(outputfile, 'w')
	f.write(str)
	f.close()

def print_story(json_obj):
	for story in json_obj['list']['story']:
		print "TITLE: " + story['title']['$text'] + '\n'
		print "DATE: " + story['storyDate']['$text'] + '\n'
		print "TEASER: " + story['teaser']['$text'] + '\n'
		if 'byline' in story:
			print "BYLINE:" + story['byline'][0]['name']['$text']
		if 'show' in story: 
			print "PROGRAM: " + story['show'][0]['program']['$text']
			print "NPR URL: " + story['link'][0]['$text']
			print "IMAGE: " + story['image'][0]['src']
		if 'caption' in story:
			print 'IMAGE CAPTION: ' + story['image'][0]['caption']['$text']
		if 'producer' in story:
			print 'IMAGE CREDIT: ' + story['image'][0]['producer']['$text']
		print 'MP3 AUDIO: ' + story['audio'][0]['format']['mp3'][0]['$text']

def get_transcripts(json_obj):
	text = ''
	for story in json_obj['list']['story']:
		text += story['id'] + "\t" 
		for paragraph in story['text']['paragraph']:
			if '$text' in paragraph:
				text += paragraph['$text'] + " " 
		text += "\n"
	return text

url_query = 'http://api.npr.org/query?apiKey=' 
url_query = url_query + key

topic = raw_input("pls input the topic id: ")
story_url = url_query + '&numResults=50&format=json&id=topic&requiredAssets=text' #1007 is science #1014 is politics

content = get_transcripts(read(story_url))
write2txt(topic + '.txt', content.encode("utf-8"))

