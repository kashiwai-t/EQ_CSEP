# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
from matplotlib import cm 
import pickle
import pandas as pd
import CSEP
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import time
import datetime
import sklearn.mixture
from matplotlib.colors import LogNorm
import matplotlib.font_manager
import sklearn.linear_model as lm

visualPath = 'visualization'

############## MAIN #####################
# RI法：グリッドごとに地震回数をカウント
# NanjoRI法：グリッドごとの地震回数を半径Sの円に入るグリッド数で平滑化

if __name__ == "__main__":
	cellSize = 0.05				# セルの大きさ（°）
	mL = 2.5					# 最小マグニチュード
	#mL = 5.5					# 最小マグニチュード
	b = 0.9              # GR（Gutenberg-Richter）のパラメータ
	#Ss = [10,30,50,100]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	#Ss = [100]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	#Ss = [30]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	Ss = [10]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	#Ss = [0]		# 平滑化パラメータ（プログラムチェック用）

	sTrainDay = '1980-01-01'	# 学習の開始日
	eTrainDay = '2016-12-31'	# 学習の終了日

	#sTrainDay = '1950-01-01'	# 学習の開始日
	#eTrainDay = '2006-12-31'	# 学習の終了日

	#sTrainDay = '2012-01-01'	# 学習の開始日 # データ量を減らして
	#eTrainDay = '2016-12-31'	# 学習の終了日 # データ量を減らして

	#sTrainDay = '2016-01-01'	# 学習の開始日 # データ量を減らして
	#eTrainDay = '2016-01-31'	# 学習の終了日 # データ量を減らして
	#eTrainDay = '2016-12-31'	# 学習の終了日 # データ量を減らして

	sTestDay = '2017-01-01'		# 評価の開始日
	eTestDay = '2017-12-31'		# 評価の終了日

	#sTestDay = '2007-01-01'		# 評価の開始日
	#eTestDay = '2007-12-31'		# 評価の終了日


	  

	# CSEPのデータクラス
	myCSEP = CSEP.Data(sTrainDay, eTrainDay, sTestDay, eTestDay, mL, b)
	
	# CSEP関東領域グリッド（lon:138.475-141.525, lat:34.475-37.025）
	# CSEPの関東地方の予測地(lon:138.50-141.50 , lat:34.50-37.00)
	lats = np.arange(34.475, 36.976, cellSize)
	lons = np.arange(138.475, 141.476, cellSize)
	
	# CSEPマグニチュード別ビン
	trainMg = np.arange(mL,9.1,0.1)
	testMg_dm = np.arange(4.0,9.1,0.1)
	testMg_y = np.arange(5.0,9.1,0.1)

	# CSEPの学習期間
	TrainTerm = myCSEP.term_cul(sTrainDay,eTrainDay)

	# 訓練データを作成
	trainData = myCSEP.Data_sel(mL, dataType='train') # trainDataの選別
	trainData = myCSEP.getDataInGrid(34.475, 138.475, 37.025, 141.525, trainData)

	# マグニチュードの発生回数
	numsRI = np.zeros([len(lats), len(lons)])			# RI法
	numsNanjoRI = np.zeros([len(lats), len(lons)])		# NanjoRI法

	# セル中心
	cellCsFlat = np.array([[lat + cellSize/2, lon + cellSize/2] for lat in lats for lon in lons])
	cellCs = np.reshape(cellCsFlat, [len(lats), len(lons), 2])
	cell_center = np.reshape(cellCsFlat,[(len(lats))*(len(lons)),2])
	
	# 訓練データのマグニチュードM>6.5の地震をかさ増し
	TrainData = myCSEP.Data_Mg(trainData)

	#x=myCSEP.exp(trainData)
	
	# ヒストグラムで緯度経度の地震頻度
	#myCSEP.plot_hist(trainData)

	# 関東のマップ生成
	#myCSEP.mapping()

	#myCSEP.draw_heatmap(trainData[['latitude','longitude']])

	t_Data = myCSEP.getDataInGrid(34.475, 138.475, 37.025, 141.525, trainData)
	#myCSEP.mapping(trainData)

	# ガウシアン可視化
	#print(trainData.shape)
	#GMM_plot = myCSEP.gaussian(trainData)
	#GMM_plot *= len(trainData)
	#myCSEP.plot_hist2(trainData,GMM_plot)
	#myCSEP.mapping_Nanjo(t_Data,GMM_plot )
	'''
	#--------------------------------------------------------------------
	# マグニチュード毎にビン詰め
	AllData = myCSEP.Data_sel(mL, dataType='all') # trainDataの選別
	# CSEP関東領域グリッド（lon:138.475-141.525, lat:34.475-37.025）
	AllDMata = myCSEP.getDataInGrid(34.475,138.475,37.025,141.525,AllData)
	TestMg=testMg_dm[:30]
	EQ_num=np.zeros(len(TestMg))
	EQ_logNum=np.zeros(len(TestMg))
	print('地震発生回数')
	for i,mg in enumerate(TestMg):
		EQ_num[i] = len(AllData[(AllData['magnitude']>=(mg-0.05))&(AllData['magnitude']<(mg+0.05))])
		EQ_logNum[i] = np.log10(len(AllData[(AllData['magnitude']>=(mg-0.05))&(AllData['magnitude']<(mg+0.05))]))
		print('マグニチュード＝{},{}回'.format(round(mg,3),EQ_num[i]))


	fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
	plt.close()
	#fig, figInds = plt.subplots(ncols=2)
	ax1 = plt.subplot2grid((2,1), (0,0))
	ax2 = plt.subplot2grid((2,1), (1,0))
	ax1.plot(TestMg,EQ_num,'.')
	ax1.plot(TestMg,EQ_num,'-')
	ax1.set_ylabel(u'地震発生回数',fontdict = {"fontproperties": fontprop},fontsize=18)


	lml = lm.LinearRegression()
	lml.fit(TestMg[:,np.newaxis],EQ_logNum[:,np.newaxis])

	ax2.plot(TestMg,EQ_logNum,'.')
	ax2.plot(TestMg,lml.predict(TestMg[:,np.newaxis]))
	x = (EQ_logNum[0] + EQ_logNum[1])/2
	y = (EQ_logNum[13] + EQ_logNum[14])/2
	z = (EQ_logNum[25] + EQ_logNum[26])/2
	plt.yticks([x,y,z], ["3000","300","30"])
	ax2.set_xlabel(u'マグニチュード',fontdict = {"fontproperties": fontprop},fontsize=18)
	ax2.set_ylabel(u'地震発生回数',fontdict = {"fontproperties": fontprop},fontsize=18)
	plt.show()

	plt.close()
	#--------------------------------------------------------------------
	'''

	'''
	fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
	trainData = myCSEP.getDataInGrid(34.475,138.475,37.025,141.525,trainData)
	t_data = trainData[['longitude','latitude','depth']]
	#t_data = np.array(trainData[['latitude','longitude','depth']])
	num = len(t_data)
	print('Depth 30>d -> {}'.format(len(t_data[t_data['depth']<30])))
	print('Depth 30<=d<60 -> {}'.format(len(t_data[(t_data['depth']>=30) & (t_data['depth']<60)])))
	print('Depth 60<=d -> {}'.format(len(t_data[t_data['depth']>60])))
	print('num = {}'.format(num))
	
	data_S = np.array(t_data[t_data['depth']<30])
	data_M = np.array(t_data[(t_data['depth']>=30) & (t_data['depth']<60)])
	data_L = np.array(t_data[t_data['depth']>60])
	
	plt.close()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in np.arange(len(data_S)):
		print(i)
		ax.scatter(data_S[i,0],data_S[i,1],-data_S[i,2],c='r',marker='.')
	ax.set_xlabel(u'緯度',fontdict = {"fontproperties": fontprop})
	ax.set_ylabel(u'経度',fontdict = {"fontproperties": fontprop})
	ax.set_zlabel(u'深度',fontdict = {"fontproperties": fontprop})
	#fullPath = os.path.join(visualPath,'depth < 30km.png')
	#plt.savefig(fullPath)
	plt.show()
	
	plt.close()			
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in np.arange(len(data_M)):
		print(i)
		ax.scatter(data_M[i,0],data_M[i,1],-data_M[i,2],c='r',marker='.')
	ax.set_xlabel(u'緯度',fontdict = {"fontproperties": fontprop})
	ax.set_ylabel(u'経度',fontdict = {"fontproperties": fontprop})
	ax.set_zlabel(u'深度',fontdict = {"fontproperties": fontprop})
	#fullPath = os.path.join(visualPath,'depth < 60km & depth >= 30km.png')
	#plt.savefig(fullPath)
	plt.show()

	plt.close()			
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in np.arange(len(data_L)):
		print(i)
		ax.scatter(data_L[i,0],data_L[i,1],-data_L[i,2],c='r',marker='.')
	ax.set_xlabel(u'緯度',fontdict = {"fontproperties": fontprop})
	ax.set_ylabel(u'経度',fontdict = {"fontproperties": fontprop})
	ax.set_zlabel(u'深度',fontdict = {"fontproperties": fontprop})
	#fullPath = os.path.join(visualPath,'60km < depth .png')
	#plt.savefig(fullPath)
	plt.show()

	ax.legend([u'地震発生場所'], prop = fontprop, loc='upper right', borderaxespad=0, fontsize=24)

	plt.show()
	'''


	#'''
	# -------------------------------------------------------------
	# plot
	plt.close()
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

	# セルの中心
	plt.scatter(cell_center[:,1],cell_center[:,0],c='g',marker='.')

	plt.legend([u'地震発生場所',u'セル中心'], prop = fontprop, loc='upper right', borderaxespad=0, fontsize=24)

	# lon:138.475-141.525, lat:34.475-37.025
	lon_min=138.475
	lon_max=141.525
	lat_min=34.475
	lat_max=37.025

	# セルごとに区切るように罫線を引く
	for lat in lats:
		p = plt.plot([lon_min,lon_max],[lat,lat],"black",linestyle='-')
		for lon in lons:
			p = plt.plot([lon,lon],[lat_min,lat_max],"black",linestyle='-')

	plt.scatter(140.90,36.300,c='b',marker='.')
	c = plt.Circle((140.90,36.300),radius=0.09,ec='y',fill=False,linewidth=4)
	ax.add_patch(c)
	print('[1000,0] = {},[1000,1] = {}'.format(cell_center[1000,0],cell_center[1000,1]))
	plt.xlabel(u'経度', fontdict = {"fontproperties": fontprop})
	plt.ylabel(u'緯度', fontdict = {"fontproperties": fontprop})
	plt.title(u'地震発生位置', fontdict = {"fontproperties": fontprop}) # グラフのタイトル
	plt.show()
	# -------------------------------------------------------------
	#'''

	#--------------------------------------------------------------------------------------
	for S in Ss:
		a=0
		print("S:{}km".format(S))
		for i, lat in enumerate(lats):
			print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				#print("longitude:{}...".format(lon))
				tmpData = myCSEP.getDataInGrid(lat, lon, lat+cellSize, lon+cellSize, trainData)
				
				#-------------
				#-----------------------------------------------------------------------------
				# RI法：各セルのマグニチュードmL以上の地震をマグニチュード毎カウント
				# セルごとにカウント
				#print(len(tmpData))
				numsRI[i,j] = len(tmpData)
				#pdb.set_trace()
				a+=len(tmpData)
				#-------------
				# Nanjo RI法：各セルのマグニチュードmL以上の地震をカウントし、
				# 距離S以内のセル全てに(Nsb+1)^-1をカウント
				
				# セル中心からS以内にセル中心があるセルのインデックスとセル数Nsbを取得
				dists = myCSEP.deg2dis(cellCs[i,j,0],cellCs[i,j,1], cellCsFlat[:,0],cellCsFlat[:,1])
				latInds,lonInds = np.where(np.reshape(dists,[(len(lats)),(len(lons))])<S)	# インデックス
				Nsb = len(latInds)	# セル数
			
				# (Nsb + 1)^-1を割り当てる
				for k, l in zip(latInds, lonInds):
					numsNanjoRI[k,l] += (numsRI[i,j]*(1/(Nsb + 1)))
				#--------------------------------------------------------------------------------------------
				
				#-------------
				# Nanjo RI法：各セルのマグニチュードmL以上の地震をカウントし、
				# 距離S以内のセル全てに(Nsb+1)^-1をカウント
				
				# セル中心からS以内にセル中心があるセルのインデックスとセル数Nsbを取得
					dists = myCSEP.deg2dis(cellCs[i,j,0],cellCs[i,j,1], cellCsFlat[:,0],cellCsFlat[:,1])
					latInds,lonInds = np.where(np.reshape(dists,[(len(lats)),(len(lons))])<S)	# インデックス
					Nsb = len(latInds)	# セル数
		
				#-------------
				#--------------------------------------------------------------------------------------------
				#numsRI[i,j] = np.sum(tmpData['magnitude'].values > mL)
			
		# カウントされた地震の保存
		numsRI_o = numsRI
					  
		EQ_num = a
		#-------------------
		# 最も良い尤度の時のSパラメータの選択

	#-------------------------------------------------------------------------------------------



		print(numsNanjoRI.shape)
		myCSEP.plot_hist2(trainData,numsNanjoRI)
		#---------------------

		#---------------------	
		# 正規化
		numsRI = numsRI / np.sum(numsRI)
		numsNanjoRI = numsNanjoRI / np.sum(numsNanjoRI)

		myCSEP.mapping_Nanjo(t_Data,numsRI)
		myCSEP.mapping_Nanjo(t_Data,numsNanjoRI)

		# heatmap で表示
		#myCSEP.draw_heatmap(numsRI)
		#myCSEP.draw_heatmap(numsNanjoRI)

		#---------------------	
		
		'''
		#---------------------
		# プロット
		plt.close()
		fig, figInds = plt.subplots(ncols=2)
		figInds[0].imshow(numsRI,cmap="bwr")
		figInds[0].set_title('Relative Intensity')

		figInds[1].imshow(numsNanjoRI,cmap="bwr")
		figInds[1].set_title('Nanjo Relative Intensity')

		fullPath = os.path.join(visualPath,'RIvsNanjoRI_{}km.png'.format(S))
		plt.show()
		plt.savefig(fullPath)
			
		#--------------------------------------------------------------------------------
		plt.close()
		n = 10
		flag=0
		fig = plt.figure()
		ax1 = fig.add_subplot(121, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')
		#ax1 = fig.add_subplot(111, projection='3d')
		for i,lat in enumerate(lats):
			for j,lon in enumerate(lons):
				if(numsRI[i,j] ==0):
					c = 'b'
					m = 'o'
					l = 'RI = 0'
				else:
					c = 'r'
					m = '^'
					l = 'RI > 0'
					flag=1
					print('numsRI({},{}) = {}'.format(i,j,numsRI[i,j]))
				ax1.scatter(lon,lat,numsRI[i,j], c = c, marker=m,label=l)
		ax1.set_xlabel('latitude')
		ax1.set_ylabel('lontitude')
		ax1.set_zlabel('predict')
		ax1.set_title('RI') # グラフのタイトル
		
		fullPath = os.path.join(visualPath,'RI.png'.format(S))
		plt.savefig(fullPath)
		
		#-----------------------------------------
		width = depth = cellSize
		pdb.set_trace()
		X, Y = np.meshgrid(lats, lons)
		x, y = X.ravel(), Y.ravel()
		RI_num = np.reshape(numsRI,x.shape)
		ax2.bar3d(x, y, np.zeros_like(numsRI), width, depth, RI_num, shade=True)

		ax2.set_xlabel('latitude')
		ax2.set_ylabel('lontitude')
		ax2.set_zlabel('predict')
		ax2.set_title('RI') # グラフのタイトル
		plt.show()

		#-----------------------------------------
		plt.close()
		flag=0
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i,lat in enumerate(lats):
			for j,lon in enumerate(lons):
				if(numsNanjoRI[i,j] ==0):
					c = 'b'
					m = 'o'
					l = 'RI = 0'
				else:
					c = 'r'
					m = '^'
					l = 'RI > 0'
					flag=1
					print('numsNanjoRI({},{}) = {}'.format(i,j,numsNanjoRI[i,j]))
				ax.scatter(lat,lon,numsNanjoRI[i,j], c = c, marker=m,label=l)
		ax.set_xlabel('latitude')
		ax.set_ylabel('lontitude')
		ax.set_title('NanjoRI {}km'.format(S)) # グラフのタイトル
		fullPath = os.path.join(visualPath,'NanjoRI_{}km.png'.format(S))
		plt.savefig(fullPath)
		#--------------------------------------------------------------------------------
		'''

	#---------------------------------------------------------------------
	# GMMによる確率密度計算
	# ガウシアン可視化
	TrainData = myCSEP.Data_Mg(TrainData)
	k_RI = myCSEP.gaussian(TrainData)
	#---------------------------------------------------------------------
	#-----------------------------------------------------------------

	Days = [1,90,365]		# 評価期間（テストデータのスタートからの日数[day]）
	#Days = [365]		# 評価期間（テストデータのスタートからの日数[day]）
	# マグニチュードの発生回数(sTrain~評価期間終了までの)
	testData_dm = myCSEP.Data_sel(4.0, dataType='test') # testDataの選別
	testData_y  = myCSEP.Data_sel(5.0, dataType='test') # testDataの選別
	train_Num = EQ_num / TrainTerm
	print(len(testData_dm))

	for k, term in enumerate(Days):
		# 予測期間による場合分け
		if term<365:# 1年以内
			testData = testData_dm # 1日、3ヶ月予測の場合
			testData = myCSEP.getDataInGrid(34.475,138.475,37.025,141.525,testData) # 緯度経度を範囲内のみ選択
			# 地震の予測回数 
			RI_estimate      = myCSEP.GR_fre(numsRI     , train_Num, testMg_dm, term)
			NanjoRI_estimate = myCSEP.GR_fre(numsNanjoRI, train_Num, testMg_dm, term)
			KashiwaiRI_estimate = myCSEP.GR_fre(k_RI, train_Num, testMg_dm, term)
			obsRI = np.zeros([len(lats), len(lons), len(testMg_dm)])

		else:# １年以上
			testData = testData_y # 1年予測の場合
			testData = myCSEP.getDataInGrid(34.475,138.475,37.025,141.525,testData) # 緯度経度を範囲内のみ選択
			# 地震の予測回数 
			RI_estimate      = myCSEP.GR_fre(numsRI     , train_Num, testMg_y, term)
			NanjoRI_estimate = myCSEP.GR_fre(numsNanjoRI, train_Num, testMg_y, term)
			KashiwaiRI_estimate = myCSEP.GR_fre(k_RI, train_Num, testMg_y, term)
			obsRI = np.zeros([len(lats), len(lons), len(testMg_y)])
			

		# マグニチュードの発生回数(sTrain~評価期間終了までの)
		tmpData = myCSEP.splitTestDataInGrid(term, testData)
		print("term: {}days".format(term))
		for i, lat in enumerate(lats):
			#print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				if (tmpData.shape!=(1,)):
					tmpDATA = myCSEP.getDataInGrid(lat, lon, lat+cellSize, lon+cellSize, tmpData)
					#-------------
					# 各セルのマグニチュードmL以上の地震をカウント
					tmpDATA = myCSEP.Data_mgInGrid(tmpDATA)
				
					if (len(tmpDATA) >= 1):
						#print('true.  {}'.format(tmpDATA))
						for num in tmpDATA:
							obsRI[i,j,int(num)] += 1
					
					#-------------
				else:
					break

		#obsRI=観測データ（トレーニングとテストの地震のカウント数を合計）
		obsRI_o = obsRI

		#obsRI = numsRI_o + obsRI
		obsRI = obsRI / np.sum(obsRI) # 正規化


		#-------------------------------
		# plot
	

		#--------------------------------

		#モデルの尤度を計算

	
		if (term<365):# 1年以内
			LL = myCSEP.likelifood(RI_estimate        , obsRI_o, lats, lons, testMg_dm, train_Num, term)
			print('LL numsRI :{}day {}'.format(term,float(LL)))
			
			LL = myCSEP.likelifood(NanjoRI_estimate   , obsRI_o, lats, lons, testMg_dm, train_Num, term)
			print('LL numsNanjoRI :{}day {}'.format(term,float(LL)))

			LL = myCSEP.likelifood(KashiwaiRI_estimate, obsRI_o, lats, lons, testMg_dm, train_Num, term)
			print('LL numsKashiwaiRI :{}day {}'.format(term,float(LL)))
		
		else:# １年以上
			LL = myCSEP.likelifood(RI_estimate        , obsRI_o, lats, lons, testMg_y, train_Num, term)
			print('LL numsRI :{}day {}'.format(term,float(LL)))
			
			LL = myCSEP.likelifood(NanjoRI_estimate   , obsRI_o, lats, lons, testMg_y, train_Num, term)
			print('LL numsNanjoRI :{}day {}'.format(term,float(LL)))

			LL = myCSEP.likelifood(KashiwaiRI_estimate, obsRI_o, lats, lons, testMg_y, train_Num, term)
			print('LL numsKashiwaiRI :{}day {}'.format(term,float(LL)))
		#---------------------	
	#-------------------------------------------------------------------------------
#########################################
#########################################
