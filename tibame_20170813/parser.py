from bs4 import BeautifulSoup

html_sample = '''
<html>
	<body>
		<h1 id="title">Hello World</h1>
		<a href="#" class= "link">This is link1</a>
		<a href="# link2" class= "link">This is link2</a>
	</body>
</html>
'''

soup = BeautifulSoup(html_sample,'lxml')
#print(soup.text) # text: remove all tag
alink = soup.select('a') # store to list
#print(alink[0])
#print(soup.select('.link'))
#print(soup.select('#title'))


# body a
#print(soup.select('body a'))


#----------
for link in soup.select('a.link'):
	print(link['href'])
