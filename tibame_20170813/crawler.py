import requests
import pandas
appledaily_url = 'http://www.appledaily.com.tw/realtimenews/section/new/'
appledaily_domain = 'http://www.appledaily.com.tw/'
thsr_url = 'https://www.thsrc.com.tw/tw/TimeTable/SearchResult'
thsr_payload = {
	'StartStation':'a7a04c89-900b-4798-95a3-c01c455622f4',
	'EndStation':'e6e26e66-7dc1-458f-b2f3-71ce65fdc95f',
	'SearchDate':'2017/08/18',
	'SearchTime':'15:30',
	'SearchWay':'DepartureInMandarin'
}
tra_url = 'http://twtraffic.tra.gov.tw/twrail/SearchResult.aspx?searchtype=0&searchdate=2017/08/18&fromstation=1810&tostation=1008&trainclass=%271100%27,%271101%27,%271102%27,%271107%27,%271110%27,%271120%27&fromtime=0600&totime=2359'

vscine_url = 'http://web.vscinemas.com.tw/vsTicketing/ticketing/search_seat.aspx?area=1\
			&film=HO00006223&time=2017%2F08%2F18\
			&number=2&from=15%3A00&to=23%3A00'

def httpGet (url):
	res = requests.get(url)
	return res.text

def httpPost(url,payload):
	res = requests.post(url,data = payload)
	return res.text


#print(httpGet(vscine_url))
#print (httpGet(appledaily_url))
#print (httpPost(thsr_url,thsr_payload))

#----------------
from bs4 import BeautifulSoup

soup = BeautifulSoup(httpGet(appledaily_url),'lxml')
#print(soup)

news_list=[]

for link in soup.select('.rtddt a'):
	dic ={}
	# print(link.select('time')[0].text)
	# print(link.select('h1')[0].text)
	# print(link.select('h2')[0].text)
	# print(appledaily_domain + link['href'])

	dic['dt'] = link.select('time')[0].text
	dic['title'] = link.select('h1')[0].text
	dic['category'] = link.select('h2')[0].text
	dic['link'] = appledaily_domain + link['href']
	news_list.append(dic)
	#print (category,dt,title,link)

df = pandas.DataFrame(news_list)
print(df.head())
df.to_csv('news.csv')