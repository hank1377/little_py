import requests
import pandas
from bs4 import BeautifulSoup

seat_url = 'https://sales.vscinemas.com.tw/VieShowTicket/Home/SelectSeats'

'ASP.NET_SessionId=njl0nmghztigqedmhh4lbqqi; _gat=1; _ga=GA1.3.25115277.1503041884; _gid=GA1.3.1366520791.1503382649'
'ASP.NET_SessionId=njl0nmghztigqedmhh4lbqqi; _gat=1; _ga=GA1.3.25115277.1503041884; _gid=GA1.3.1366520791.1503382649'
headers = {
'Cache-Control':'max-age=0',
'Connection':'keep-alive',
'Cookie':'ASP.NET_SessionId=njl0nmghztigqedmhh4lbqqi; _gat=1; _ga=GA1.3.25115277.1503041884; _gid=GA1.3.1366520791.1503382649',
'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36'
}

res = requests.post(seat_url,data={},headers=headers)

soup = BeautifulSoup(res.text,'html.parser').select('td[data-type="Empty"]')

seat_ary=[]




for seat in soup:
	seat_ary.append({'Seat_no':seat['data-col'],'Seat_row':seat['data-name']})

df = pandas.DataFrame(seat_ary)

print(df)