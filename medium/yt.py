from pytube import YouTube
from pprint import pprint

#yturl = 'https://www.youtube.com/watch?v=bgCcnPC098o&feature=em-share_video_user'
yturl = 'https://www.youtube.com/embed/bgCcnPC098o'

downloadPath = '/home/hank/little_pro/'

def youtubeCrab(yturl,downloadPath):
	yt = YouTube(yturl)
	yt.private = True
	#pprint(yt.get_videos())
	print(yt.filename)
	yt.set_filename('myFirstVideo')
	#pprint(yt.filter('flv'))
	print(yt.filter('.mp4'))
	#pprint(yt.filter(resolution='480p'))
	#video = yt.get('mp4','360p')
	#video = yt.filter('.mp4')[-1]
	#video.download(downloadPath)

youtubeCrab(yturl,downloadPath)