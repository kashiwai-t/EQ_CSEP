# -*- ooding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt 
import pickle
import pandas as pd
import pandas.tseries.offsets as offsets
import math
import datetime
from sklearn.mixture import GaussianMixture as GMM
import test
import sklearn.mixture
from matplotlib.colors import LogNorm
import matplotlib.font_manager
from scipy import stats
from sklearn.cluster import KMeans
import conda
import os
from matplotlib.ticker import NullFormatter
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import plotly.plotly as py
import plotly.graph_objs as go


conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib


from mpl_toolkits.basemap import Basemap

visualPath = 'visualization'

#########################################


class Data:

	#--------------------------
	# データの読み込み
	def __init__(self, sTrain, eTrain, sTest, eTest, mL, b, dataPath='data'):
		self.sTrain = sTrain
		self.eTrain = eTrain
		self.sTest = sTest
		self.eTest = eTest
		self.dataPath = dataPath
		self.mL = mL
		self.b = b

		fullPath = os.path.join(self.dataPath,'atr.dat')
		self.data = pd.read_csv(fullPath,sep='\t',index_col='date', parse_dates=['date'])
		
		# 学習データ
		self.dataTrain = self.data[sTrain:eTrain]
		
		# テストデータ
		self.dataTest = self.data[sTest:eTest]
		
	#--------------------------
	#--------------------------
	def term_cul (self,sTday,eTday):
		eDay = datetime.date(int(eTday[0:4]),int(eTday[5:7]),int(eTday[8:10]))
		sDay = datetime.date(int(sTday[0:4]),int(sTday[5:7]),int(sTday[8:10]))

		e_s_days = (eDay-sDay).days

		return e_s_days


	#--------------------------
	# GR則
	# data :ここでは、過去のデータをもとに作成したデータを用いる
	#（観測回数＊テスト期間/観測期間）＊（正規化した地震回数）= （テスト期間の地震回数の予測値）
	# model:RI or NanjoRI での確率分布
	# M    :マグニチュード
	# term :地震予測の期間（１日〜３６５日）

	def GR_fre(self, model, Num_Day, Mg, term):
		
		data = Num_Day * term * model
		b_low  = np.zeros([len(model[:,0]),len(model[0]),len(Mg)])
		b_high = np.zeros([len(model[:,0]),len(model[0]),len(Mg)])
		for i, MG in enumerate(Mg):
			Mg_low = MG - 0.05 # M1
			Mg_high  = MG + 0.05 # M2

			b_low[:,:,i]  = np.log(data) - self.b * (Mg_low  - self.mL)
			b_high[:,:,i] = np.log(data) - self.b * (Mg_high - self.mL)


		d_num = np.power(10,b_low) - np.power(10,b_high)
		return d_num


	#--------------------------
	#GMM（ガウシアンミクスチャーモデリング）
	def gaussian(self,data):
		#pdb.set_trace()
		#----------------------------------------------
		# データの混合ガウスモデル
		t_Data = self.getDataInGrid(34.475, 138.475, 37.025, 141.525, data)
		t_data = np.array(t_Data[['latitude','longitude']])
		#t_data = np.array(t_Data[['latitude','longitude','magnitude']])
		#t_data = np.array(t_Data[['latitude','longitude','depth','magnitude']])
		# CSEPの関東地方の予測点(lon:138.50-141.50 , lat:34.50-37.00)
		y = np.linspace(138.48,141.52,61*5)
		x = np.linspace(34.48,37.02,51*5)
		#z = np.linspace(0,100,3)
		#y = np.linspace(138.50,141.50,61)
		#x = np.linspace(34.50,37.00,51)
		#w = np.linspace(4.0,9.0,51)
		X, Y = np.meshgrid(x, y)
		#X, Y,  W = np.meshgrid(x, y,  w)
		#X, Y, Z, W = np.meshgrid(x, y, z, w)
		XX = np.array([X.ravel(), Y.ravel()]).T
		#XX = np.array([X.ravel(), Y.ravel(), W.ravel()]).T
		#XX = np.array([X.ravel(), Y.ravel(), Z.ravel(), W.ravel()]).T

		# n-components決定
		#n_com = np.array([10])
		n_com = np.array([122])
		#n_com = np.array([70, 80, 90, 100, 110, 120])
		#n_com = np.array([10, 122])
		#n_com = np.arange(120,140)
		#n_com = np.arange(10,20)
		n_bic = np.zeros(len(n_com))
		print(t_data.shape)
		'''
		n_com = np.array([70, 80, 90, 100, 110, 120])

			clf = sklearn.mixture.GMM(N__, covariance_type='full')
			clf.fit(t_data)
			NN[i] = clf.bic(t_data)
		print(NN)
		'''
		i=0
		for i,n_components in enumerate(n_com):
			plt.close()
			#clf = sklearn.mixture.GMM(n_components, covariance_type='full').fit(t_data)
			clf = sklearn.mixture.GaussianMixture(n_components, covariance_type='full')
			clf.fit(t_data)
			n_bic[i] = clf.bic(t_data)
			# 結果を表示
			print("components = {}".format(n_components))

			print("*** weights")
			print(clf.weights_)
		
			print("*** means")
			print(clf.means_)
		
			print("*** covars")
			print(clf.covariances_)
		
			print("*** BIC")
			print(clf.bic(t_data))
			#print(pd.DataFrame(clf.predict(t_data)))
			'''
			for k in range(n_components):
				# 平均を描画
				plt.plot(clf.means_[k][1], clf.means_[k][0], 'ro')
			
				# ガウス分布の等高線を描画
				#P = plt.bivariate_normal(X, Y, np.sqrt(clf.covars_[k][1][1]), np.sqrt(clf.covars_[k][0][0]), clf.means_[k][1], clf.means_[k][0], clf.covars_[k][1][0])
				#P = plt.bivariate_normal(Y, X, np.sqrt(clf.covariances_[k][0][0]), np.sqrt(clf.covariances_[k][1][1]), clf.means_[k][0], clf.means_[k][1], clf.covariances_[k][0][1])
			Z = clf.score_samples(XX)
			Z = Z.reshape(X.shape)
			fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
			CS = plt.contour(Y, X, Z, level=10 )
			#CB = plt.colorbar(CS, shrink=0.8, extend='both')
			CB = plt.colorbar(CS)
			plt.scatter(t_data[:, 1], t_data[:, 0], .8,alpha=0.5)
			
			plt.title(u'混合ガウスモデルの地震発生頻度（対数尤度）',fontdict = {"fontproperties": fontprop},fontsize=18)
			plt.xlabel(u'緯度',fontdict = {"fontproperties": fontprop},fontsize=18)
			plt.ylabel(u'経度',fontdict = {"fontproperties": fontprop},fontsize=18)

			plt.axis('tight')
			fullPath = os.path.join(visualPath,'CSEP_GMM, n_components = {}.png'.format(n_components))
			#plt.savefig(fullPath)
			plt.show()
			'''
			#i += 1
			#print(NN)
		'''
		plt.plot(np.arange(120,140),n_bic)

		fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
		plt.title(u'BICの最尤推定',fontdict = {"fontproperties": fontprop},fontsize=25)
		plt.xlabel(u'クラスター数',fontdict = {"fontproperties": fontprop},fontsize=25)
		plt.ylabel(u'BIC',fontdict = {"fontproperties": fontprop},fontsize=25)
		plt.ylim([np.min(n_bic),np.max(n_bic)])
		plt.show()
		'''
		#pdb.set_trace()
		logProb = clf.score_samples(XX)
		Prob_s = np.exp( logProb ).reshape(X.shape)
		#Prob_s = Prob_s.reshape(X.shape)
		'''
		Prob = np.zeros([61,51])
		for i in np.arange(61):
			for j in np.arange(51):
				Prob[i,j] = Prob_s[i,j,:].sum()
		'''
		print(n_bic)
		#logProb = clf.score_samples(XX)
		#Prob = np.zeros( X.shape )
		Prob = np.zeros([61, 51])
		#Prob = np.exp( logProb.reshape(X.shape) )

		for i in np.arange(61):
			for j in np.arange(51):
				A = Prob_s[i*5  ,j*5] + Prob_s[i*5  ,j*5+1] + Prob_s[i*5  ,j*5+2] + Prob_s[i*5  ,j*5+3] + Prob_s[i*5  ,j*5+4] 
				B = Prob_s[i*5+1,j*5] + Prob_s[i*5+1,j*5+1] + Prob_s[i*5+1,j*5+2] + Prob_s[i*5+1,j*5+3] + Prob_s[i*5+1,j*5+4] 
				C = Prob_s[i*5+2,j*5] + Prob_s[i*5+2,j*5+1] + Prob_s[i*5+2,j*5+2] + Prob_s[i*5+2,j*5+3] + Prob_s[i*5+2,j*5+4] 
				D = Prob_s[i*5+3,j*5] + Prob_s[i*5+3,j*5+1] + Prob_s[i*5+3,j*5+2] + Prob_s[i*5+3,j*5+3] + Prob_s[i*5+3,j*5+4] 
				E = Prob_s[i*5+4,j*5] + Prob_s[i*5+4,j*5+1] + Prob_s[i*5+4,j*5+2] + Prob_s[i*5+4,j*5+3] + Prob_s[i*5+4,j*5+4] 
				Prob[i,j] = A + B + C + D + E
		#print(np.sum(Prob)/20/20)
		#return Prob.reshape(X.shape).T/80/80
		print(Prob.sum()/100/100)
		#return Prob.T/20/20
		return Prob.T/100/100

	#-------------------

	#--------------------------
	# Train_Num:学習データの一日の平均地震回数（回数/日）
	# obs      :テストデータの観測回数

	def Poisson(self, model, obs, Train_Num, TestTerm):
		# 尤度計算用、ポアソン
		ave = model*(TestTerm*Train_Num)

		den = 1/(math.factorial(int(obs)))

		# ポアソン分布の計算
		poi = (pow(ave,obs)*pow(math.e,(-ave)))*den

		return poi

	#--------------------------

	#-------------------------------------------
	# 尤度関数
	# ポアソン分布を用いて尤度の決定
	# 確率密度の積

	def likelifood(self, model, obs, lats, lons, Mg, Train_Num, TestTerm):
		LL=1.0
		flag = False

		tmpLL = np.zeros([len(lats), len(lons), len(Mg)])
		P = np.zeros([len(lats), len(lons), len(Mg)])


		for i, lat in enumerate(lats):
			#print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				for k, mg in enumerate(Mg):
					if model[i][j][k] == 0:
						if obs[i][j][k] == 0:
							tmpLL[i][j][k] = 1.0
							P[i][j][k] = 1.0
						if obs[i][j][k] > 0:
							tmpLL[i][j][k] = 0
							P[i][j][k] = 0
							flag = True
							break
					else:
						#tmplogLL対数尤度の計算
						tmpLL[i][j][k] = -1*model[i][j][k] + obs[i][j][k] *  math.log(model[i][j][k]) - math.log(math.factorial(obs[i][j][k]))
						# 確率（ポアソン分布）
						P[i][j][k] = self.Poisson(model[i][j][k],obs[i][j][k], Train_Num, TestTerm) # 尤度
					if flag:
						break
						
					
		
		LL = np.sum(tmpLL) # 対数尤度
	
		return LL
		#return -math.log(LL)


	#-----------------------------------------------
	#-----------------------------------------------
	# データタイプを選択し、そのデータを返す
	# dataType: all, train, test
	# all  :気象庁データの全て（1923〜2016）
	# train:学習データ
	# test :テストデータ

	def Data_sel(self, mL, dataType='train'):
		if dataType=='all':
			data = self.data
		elif dataType=='train':
			data = self.dataTrain
		elif dataType=='test':
			data = self.dataTest
		Data = data[data['magnitude'].values > (mL-0.05)]	
		
		return Data

	#------------------------------------------------
	#------------------------------------------------
	# Test期間に応じてテスト開始日からのデータを抽出

	def splitTestDataInGrid(self , term, data):
		date = datetime.datetime.strptime(self.sTest, '%Y-%m-%d') + datetime.timedelta(term)
		datestr = date.strftime("%Y-%m-%d")
		tmpData = data[(data.index >= self.sTest) & (data.index <= datestr)]
		if (len(tmpData) == 0 ):
			tmpData = np.arange(1)
		return tmpData

	#------------------------------------------------
	#------------------------------------------------
	# マグニチュードを0.1刻みでビン分割

	def Data_mgInGrid(self, data):

		# magnitudeの強さだけの配列
		D_mg = np.array(data['magnitude'])

		# D_mgは何番目のビンかを決める
		for i ,D in enumerate(D_mg):
			D_mg[i] = round((D-self.mL)*10,1) 


		return D_mg


	#------------------------------------------------

	#------------------------------------------------
	# マグニチュードM>6.5で地震頻度を増す

	def Data_Mg(self, data):

		# magnitudeの強さによって、かさ増し
		D_mg = data[data['magnitude']>6.5]

		# dataを連結させる
		D_mg = D_mg.append(data) 
		D_mg = D_mg.append(data) 
		D_mg = D_mg.append(data) 

		return D_mg
	#------------------------------------------------

	#--------------------------
	# グリッド内のデータ取り出し
	# sLat: 開始緯度
	# sLon: 開始経度
	# eLat: 終了緯度
	# eLon: 終了経度
	def getDataInGrid(self, sLat, sLon, eLat, eLon, Data):
		tmpData = Data[(Data['latitude'] >= sLat) & (Data['latitude'] < eLat) & (Data['longitude'] >= sLon)  & (Data['longitude'] < eLon)]
		

		return tmpData
	#--------------------------

	
	#--------------------------
	# sliding windowでデータを分割
	# winIn: 入力用のウィンドウ幅（単位：月）
	# winOut: 出力用のウィンドウ幅（単位：月）	
	# stride: ずらし幅（単位：月）
	def splitData2Slice(self, winIn=120, winOut=3, stride=1):
	
		# ウィンドウ幅と、ずらし幅のoffset
		winInOffset = offsets.DateOffset(months=winIn, days=-1)
		winOutOffset = offsets.DateOffset(months=winOut, days=-1)
		strideOffset = offsets.DateOffset(months=stride)
		
		# 学習データの開始・終了のdatetime
		sTrainDT = pd.to_datetime(self.sTrain)
		eTrainDT = pd.to_datetime(self.eTrain)
		
		#---------------
		# 各ウィンドウのdataframeを取得
		self.dfX = []
		self.dfY = []
		
		# 現在の日時
		currentDT = sTrainDT
		endDTList = [] # Saito temporarily added (7/9)
		while currentDT + winInOffset + winOutOffset <= eTrainDT:
			endDTList.append(currentDT+winInOffset) # Saito temporarily added (7/9)
		
			# 現在の日時からwinInOffset分を抽出
			self.dfX.append(self.dataTrain[currentDT:currentDT+winInOffset])

			# 現在の日時からwinInOffset分を抽出
			self.dfY.append(self.dataTrain[currentDT+winInOffset:currentDT+winInOffset+winOutOffset])

			# 現在の日時をstrideOffset分ずらす
			currentDT = currentDT + strideOffset
		#---------------
        
		return self.dfX, self.dfY, endDTList, # Saito temporarily added (7/9)
	#--------------------------

	#--------------------------
	# pointCNN用のデータ作成
	def makePointCNNData(self, trainRatio=0.8):
		# 学習データとテストデータ数
		self.nData = len(self.dfX)
		self.nTrain = np.floor(self.nData * trainRatio).astype(int)
		self.nTest = self.nData - self.nTrain
		
		# ランダムにインデックスをシャッフル
		self.randInd = np.random.permutation(self.nData)
		
		
		# 学習データ
		self.xTrain = self.dfX[self.randInd[0:self.nTrain]]
		self.yTrain = self.dfY[self.randInd[0:self.nTrain]]

		# 評価データ
		self.xTest = self.dfX[self.randInd[self.nTrain:]]
		self.yTest = self.dfY[self.randInd[self.nTrain:]]
		
		
		# ミニバッチの初期化
		self.batchCnt = 0
		self.batchRandInd = np.random.permutation(self.nTrain)
		#--------------------		
	#--------------------------
	
	#------------------------------------
	# pointCNN用のミニバッチの取り出し
	def nextPointCNNBatch(self,batchSize):

		sInd = batchSize * self.batchCnt
		eInd = sInd + batchSize
		'''
		batchX = []
		batchY = []
		'''
		batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
		batchY = self.yTrain[self.batchRandInd[sInd:eInd]]
		
		
		if eInd+batchSize > self.nTrain:
			self.batchCnt = 0
		else:
			self.batchCnt += 1

		return batchX, batchY
	#------------------------------------
		
	#--------------------------
	# ヒュベニの公式を用いた緯度・経度座標系の2点間の距離(km)
	# https://qiita.com/chiyoyo/items/b10bd3864f3ce5c56291
	# を参考にして作成
	# lat1: 1点目の緯度
	# lon1: 1点目の経度
	# lat2: 2点目の緯度
	# lon2: 2点目の経度	
	# mode: 測地系の切り替え
	def deg2dis(self, lat1, lon1, lat2, lon2, mode=True):
		#lat2 = data['latitude'].values
		#lon2 = data['longitude'].values
		
		# 緯度経度をラジアンに変換
		radLat1 = lat1/180*np.pi # 緯度１
		radLon1 = lon1/180*np.pi # 経度１
		radLat2 = lat2/180*np.pi # 緯度２
		radLon2 = lon2/180*np.pi # 経度２
		
		# 緯度差
		radLatDiff = radLat1 - radLat2

		# 経度差算
		radLonDiff = radLon1 - radLon2;

		# 平均緯度
		radLatAve = (radLat1 + radLat2) / 2.0

		# 測地系による値の違い
		a = [6378137.0 if mode else 6377397.155][0]						# 赤道半径
		b = [6356752.314140356 if mode else 6356078.963][0]				# 極半径
		e2 = [0.00669438002301188 if mode else 0.00667436061028297][0]	# 第一離心率^2
		a1e2 = [6335439.32708317 if mode else 6334832.10663254][0]		# 赤道上の子午線曲率半径

		sinLat = np.sin(radLatAve)
		W2 = 1.0 - e2 * (sinLat**2)
		M = a1e2 / (np.sqrt(W2)*W2)		# 子午線曲率半径M
		N = a / np.sqrt(W2)				# 卯酉線曲率半径

		t1 = M * radLatDiff;
		t2 = N * np.cos(radLatAve) * radLonDiff
		dist = np.sqrt((t1 * t1) + (t2 * t2))

		return dist/1000
	#--------------------------

	# 日本地図のマップ生成
	def mapping(self,data):
		plt.close()
		m = Basemap(projection='merc',
			resolution='h',
			llcrnrlon=138,
			llcrnrlat=34,
			urcrnrlon=142,
			urcrnrlat=37.5)
		m.drawcoastlines(color='gray')
		m.drawcountries(color='gray')
		m.fillcontinents(color='white', lake_color='#eeeeee')
		m.drawmapboundary(fill_color='#eeeeee')
		# 5度ごとに緯度線を描く
		m.drawparallels(np.arange(34, 37.6, 0.5), labels = [1, 0, 0, 0], fontsize=10, color='white')

		# 5度ごとに経度線を描く
		m.drawmeridians(np.arange(138, 142.5, 1), labels = [0, 0, 0, 1], fontsize=10, color='white')

		pdb.set_trace()
		# 地震予測範囲のグリッド線描画
		34.475, 138.475, 37.025, 141.525
		x = np.linspace(138.475,141.525,61)
		y = np.linspace(34.475,37.025,51)
		for i,X in enumerate(x):		
			for j,Y in enumerate(y):		
				X1,Y1 = m(X,Y)
				m.plot(X1,Y1,'+',markersize=5,color='black')
				print('grid')
				print(X,Y)
		t_data = np.array(data[['longitude','latitude']])
		pdb.set_trace()
		for i in np.arange(len(t_data)):
			X, Y = m(t_data[i,1],t_data[i,0])
			m.plot(X, Y, '.', markersize=1.0, color='red')
			print('scatter')
			print(i)

		plt.show()	


	def mapping_Nanjo(self, data, Nanjo):
		plt.close()
		m = Basemap(projection='merc',
			resolution='h',
			llcrnrlon=138,
			llcrnrlat=34,
			urcrnrlon=142,
			urcrnrlat=37.5)
		m.drawcoastlines(color='gray')
		m.drawcountries(color='gray')
		#m.shadedrelief(scale=0.5)

		m.drawmapboundary(fill_color='#eeeeee')

		#'''
		# 0.5度ごとに緯度線を描く
		m.drawparallels(np.arange(34, 37.6, 0.5), labels = [1, 0, 0, 0], fontsize=10, color='white')
		# 0.5度ごとに経度線を描く
		m.drawmeridians(np.arange(138, 142, 0.5), labels = [0, 0, 0, 1], fontsize=10, color='white')
		#'''
		fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
		t_data = np.array(data[['longitude','latitude']])
		'''
		# 地震発生場所のplot
		for i in np.arange(len(t_data)):
			X, Y = m(t_data[i,0],t_data[i,1])
			m.plot(X, Y, '.', markersize=1.0, color='red')
			print(i)

		# セルの中心
		x = np.linspace(138.475,141.525,61)
		y = np.linspace(34.475,37.025,51)
		for i,X in enumerate(x):		
			for j,Y in enumerate(y):		
				X1, Y1 = m(x[i]+0.025,y[j]+0.025)	
				m.plot(X1, Y1, '.', markersize=1, color='black')

		plt.legend([u'地震発生場所',u'セル中心'], prop = fontprop, loc='upper right', borderaxespad=0, fontsize=24)

		pdb.set_trace()
		# セル毎に区切る十字線
		for i,X in enumerate(x):		
			for j,Y in enumerate(y):		
				X1,Y1 = m(X,Y)
				m.plot(X1,Y1,'+',markersize=5,color='black')
			print(X)
		'''
		
		'''
		#-----------------------------------
		# heatmap
		x = np.linspace(138.475, 141.525,62)
		y = np.linspace(34.475, 37.025,52)
		#x = np.linspace(138.50, 141.00,61)
		#y = np.linspace(34.50, 37.00,51)
		lon, lat = np.meshgrid(x, y)
		X, Y = m(lon, lat)
		m.pcolormesh(lon, lat, Nanjo*100,
		             latlon=True, cmap='bwr')
		plt.clim(0, Nanjo.mean()*150)
		plt.colorbar()
		#-----------------------------------
		'''
		X, Y = m(140,35.5)
		plt.Circle((X, Y),radius=0.09,ec='y',fill=False,linewidth=4)
		pdb.set_trace()
		#m.contour(X, Y, Nanjo)

		#plt.legend([u'地震発生確率(%)'], prop = fontprop, borderaxespad=0, fontsize=24)

		pdb.set_trace()
		#X, Y = m(140.90,36.300)
		#m.drawgreatcircle(X ,Y ,X ,Y)
		plt.show()	
		













		'''
		ax = plt.axes()
		fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
		trainData = myCSEP.getDataInGrid(34.475,138.475,37.025,141.525,trainData)
		t_data = np.array(trainData['latitude'])
		num = len(t_data)
		t_data = np.append(t_data,trainData['longitude']).reshape(2,num)
		t_data = t_data.T
		y = np.linspace(34.475, 37.025,len(lats))
		x = np.linspace(138.475, 141.525,len(lons))
		print(x,y)
		X, Y = np.meshgrid(y, x)
		plt.grid()
		plt.scatter(t_data[:,1],t_data[:,0],c='r',marker='.')
		
		
		# lon:138.475-141.525, lat:34.475-37.025
		lon_min=138.475
		lon_max=141.525
		lat_min=34.475
		lat_max=37.025
		
		
		plt.scatter(140.90,36.300,c='b',marker='.')
		c = plt.Circle((140.90,36.300),radius=0.09,ec='y',fill=False,linewidth=4)
		ax.add_patch(c)
		print('[1000,0] = {},[1000,1] = {}'.format(cell_center[1000,0],cell_center[1000,1]))
		plt.xlabel(u'経度', fontdict = {"fontproperties": fontprop})
		plt.ylabel(u'緯度', fontdict = {"fontproperties": fontprop})
		plt.title(u'地震発生位置', fontdict = {"fontproperties": fontprop}) # グラフのタイトル
		'''




	#--------------------------
	def plot_hist(self, data, dataset=False):

		plt.close()
		t_Data = self.getDataInGrid(34.475, 138.475, 37.025, 141.525, data)
		t_data = np.array(t_Data[['latitude','longitude']])
		
		nullfmt = NullFormatter()         # no labels
		
		
		fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
		# definitions for the axes
		left, width = 0.1, 0.65
		bottom, height = 0.1, 0.65
		bottom_h = left_h = left + width + 0.02
		
		rect_scatter = [left, bottom, width, height]
		rect_histx = [left, bottom_h, width, 0.2]
		rect_histy = [left_h, bottom, 0.2, height]
		
		# start with a rectangular Figure
		plt.figure(1, figsize=(8, 8))
		
		axScatter = plt.axes(rect_scatter)
		axHistx = plt.axes(rect_histx)
		axHisty = plt.axes(rect_histy)
		
		# no labels
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)
		
		# the scatter plot:
		axScatter.scatter(t_data[:,1],t_data[:,0])
		axScatter.set_xlabel(u'経度',fontdict = {"fontproperties": fontprop},fontsize=20)
		axScatter.set_ylabel(u'緯度',fontdict = {"fontproperties": fontprop},fontsize=20)
		
		# now determine nice limits by hand:
		binwidth = 0.05
		xymax = max(np.max(np.abs(t_data[:,1])), np.max(np.abs(t_data[:,0])))
		lim = (int(xymax/binwidth) + 1) * binwidth
		
		axScatter.set_ylim((34.475, 37.025))
		axScatter.set_xlim((138.475, 141.525))
		
		bins = np.arange(-lim, lim + binwidth, binwidth)
		axHistx.hist(t_data[:,1], bins=bins)
		axHisty.hist(t_data[:,0], bins=bins, orientation='horizontal')
		
		axHistx.set_ylabel(u'地震発生回数',fontdict = {"fontproperties": fontprop},fontsize=20)
		axHistx.set_xlabel(u'経度',fontdict = {"fontproperties": fontprop},fontsize=20)
		
		axHisty.set_xlabel(u'地震発生回数',fontdict = {"fontproperties": fontprop},fontsize=20)
		axHisty.set_ylabel(u'緯度',fontdict = {"fontproperties": fontprop},fontsize=20)
		
		
		axHistx.set_label('a')
		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())
		
		plt.show()


	def plot_hist2(self, data, Nanjo):
		
		plt.close()
		t_Data = self.getDataInGrid(34.475, 138.475, 37.025, 141.525, data)
		t_data = np.array(t_Data[['latitude','longitude']])
		
		nullfmt = NullFormatter()         # no labels
		
		
		fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
		# definitions for the axes
		left, width = 0.1, 0.65
		bottom, height = 0.1, 0.65
		bottom_h = left_h = left + width + 0.02
		
		rect_scatter = [left, bottom, width, height]
		rect_histx = [left, bottom_h, width, 0.2]
		rect_histy = [left_h, bottom, 0.2, height]
		
		# start with a rectangular Figure
		plt.figure(1, figsize=(8, 8))
		
		axScatter = plt.axes(rect_scatter)
		axHistx = plt.axes(rect_histx)
		axHisty = plt.axes(rect_histy)
		
		# no labels
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)
		
		# the scatter plot:
		axScatter.scatter(t_data[:,1],t_data[:,0])
		axScatter.set_xlabel(u'経度',fontdict = {"fontproperties": fontprop},fontsize=20)
		axScatter.set_ylabel(u'緯度',fontdict = {"fontproperties": fontprop},fontsize=20)
		
		# now determine nice limits by hand:
		binwidth = 0.05
		xymax = max(np.max(np.abs(t_data[:,1])), np.max(np.abs(t_data[:,0])))
		lim = (int(xymax/binwidth) + 1) * binwidth
		
		axScatter.set_ylim((34.475, 37.025))
		axScatter.set_xlim((138.475, 141.525))
		
		bins = np.arange(-lim, lim + binwidth, binwidth)

		hist_l =0 
		for i in np.arange(Nanjo.shape[0]):
			hist_l = Nanjo[i,:].sum()
			if i==0:
				hist_lat = np.ones(int(hist_l))*(34.5+0.05*i)
			else:
				hist_lat = np.append(hist_lat,np.ones(int(hist_l))*(34.5+0.05*i))

		hist_l =0 
		for i in np.arange(Nanjo.shape[1]):
			hist_l = Nanjo[:,i].sum()
			if i==0:
				hist_lon = np.ones(int(hist_l))*(138.5+0.05*i)
			else:
				hist_lon = np.append(hist_lon,np.ones(int(hist_l))*(138.5+0.05*i))
		


		axHistx.hist(hist_lon, bins=bins)
		axHisty.hist(hist_lat, bins=bins, orientation='horizontal')
		#axHistx.hist(t_data[:,1], bins=bins)
		#axHisty.hist(t_data[:,0], bins=bins, orientation='horizontal')
		
		axHistx.set_ylabel(u'地震発生回数',fontdict = {"fontproperties": fontprop},fontsize=20)
		axHistx.set_xlabel(u'経度',fontdict = {"fontproperties": fontprop},fontsize=20)
		
		axHisty.set_xlabel(u'地震発生回数',fontdict = {"fontproperties": fontprop},fontsize=20)
		axHisty.set_ylabel(u'緯度',fontdict = {"fontproperties": fontprop},fontsize=20)
		
		
		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())
		
		plt.show()
			

		#-----------------------------------------------------------------------

	def draw_heatmap(self, data):
	  
		pdb.set_trace()
		plt.imshow(data,cmap="bwr")
		#sns.set()
		# ヒートマップを出力
		#sns.heatmap(data,annot=True)
		plt.show()

		#-----------------------------------------------------------------------


#########################################
