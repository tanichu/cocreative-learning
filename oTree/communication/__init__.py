from otree.api import *
import numpy as np
#from scipy.stats import wishart # 多次元ガウス分布, ウィシャート分布, ディリクレ分布
import torch
from torch.distributions import Wishart, Categorical,Dirichlet,MultivariateNormal,Multinomial


doc = """
Your app description
"""








class C(BaseConstants):
    NAME_IN_URL = 'communication'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 201 #本番は201


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
        
    #str_test=models.StringField()
    
    
    my_sign = models.StringField(initial="99")
    com_sign = models.StringField(initial="99")

    my_accept=models.BooleanField()
    com_accept=models.BooleanField()

    #人のサイン
    img0_sign = models.StringField(initial="99")
    img1_sign = models.StringField(initial="99")
    img2_sign = models.StringField(initial="99")
    img3_sign = models.StringField(initial="99")
    img4_sign = models.StringField(initial="99")
    img5_sign = models.StringField(initial="99")
    img6_sign = models.StringField(initial="99")
    img7_sign = models.StringField(initial="99")
    img8_sign = models.StringField(initial="99")
    img9_sign = models.StringField(initial="99")

    #コンピュータのカテゴリ
    com_img0_cat = models.StringField(initial="99")
    com_img1_cat = models.StringField(initial="99")
    com_img2_cat = models.StringField(initial="99")
    com_img3_cat = models.StringField(initial="99")
    com_img4_cat = models.StringField(initial="99")
    com_img5_cat = models.StringField(initial="99")
    com_img6_cat = models.StringField(initial="99")
    com_img7_cat = models.StringField(initial="99")
    com_img8_cat = models.StringField(initial="99")
    com_img9_cat = models.StringField(initial="99")


    #コンピュータのサイン
    com_img0_sign = models.StringField(initial="99")
    com_img1_sign = models.StringField(initial="99")
    com_img2_sign = models.StringField(initial="99")
    com_img3_sign = models.StringField(initial="99")
    com_img4_sign = models.StringField(initial="99")
    com_img5_sign = models.StringField(initial="99")
    com_img6_sign = models.StringField(initial="99")
    com_img7_sign = models.StringField(initial="99")
    com_img8_sign = models.StringField(initial="99")
    com_img9_sign = models.StringField(initial="99")

    #コンピュータΘ (行列のフィールドがないから分割)
    com_sita_lk_00= models.FloatField(initial=0.0)
    com_sita_lk_01= models.FloatField(initial=0.0)
    com_sita_lk_02= models.FloatField(initial=0.0)

    com_sita_lk_10= models.FloatField(initial=0.0)
    com_sita_lk_11= models.FloatField(initial=0.0)
    com_sita_lk_12= models.FloatField(initial=0.0)

    com_sita_lk_20= models.FloatField(initial=0.0)
    com_sita_lk_21= models.FloatField(initial=0.0)
    com_sita_lk_22= models.FloatField(initial=0.0)

    #分類の順番保持
    order_A0=models.StringField(initial="99")
    order_A1=models.StringField(initial="99")
    order_A2=models.StringField(initial="99")
    order_A3=models.StringField(initial="99")
    order_A4=models.StringField(initial="99")
    order_A5=models.StringField(initial="99")
    order_A6=models.StringField(initial="99")
    order_A7=models.StringField(initial="99")
    order_A8=models.StringField(initial="99")
    order_A9=models.StringField(initial="99")

    order_B0=models.StringField(initial="99")
    order_B1=models.StringField(initial="99")
    order_B2=models.StringField(initial="99")
    order_B3=models.StringField(initial="99")
    order_B4=models.StringField(initial="99")
    order_B5=models.StringField(initial="99")
    order_B6=models.StringField(initial="99")
    order_B7=models.StringField(initial="99")
    order_B8=models.StringField(initial="99")
    order_B9=models.StringField(initial="99")

    order_C0=models.StringField(initial="99")
    order_C1=models.StringField(initial="99")
    order_C2=models.StringField(initial="99")
    order_C3=models.StringField(initial="99")
    order_C4=models.StringField(initial="99")
    order_C5=models.StringField(initial="99")
    order_C6=models.StringField(initial="99")
    order_C7=models.StringField(initial="99")
    order_C8=models.StringField(initial="99")
    order_C9=models.StringField(initial="99")
    
    r= models.FloatField(initial=0.0)

    com_model=models.StringField(initial="99")


    
#inter-GMの関数--------------------------------------------------
def init_com_paramater(player):
    player.participant.com_model=player.id_in_group%3
    player.com_model=str(player.participant.com_model)
    #player.participant.AAA=1
    
    #定数とハイパーパラメータ---------------------------------------------
    D=3
    K=3
    L=3
    N=10
    inter_I=100
    xB =np.array([
        [ 61.37168596,  26.15860931, -52.10096281],
        [ 68.27001472, -38.4612657 ,  53.77227298],
        [ 53.03117157, -64.87850309,  44.4914579 ],
        [ 59.5374853 , -63.855726  ,  53.98879415],
        [ 67.11591845, -54.94274545,  53.09209315],
        [ 64.58325618,  38.28864496, -51.37276997],
        [ 62.1620156 , -40.7310968 ,  51.12112248],
        [ 63.16385962,  52.90467898, -64.3546551 ],
        [ 62.34665201,  41.46976689, -22.88672083],
        [ 67.04438997, -65.46085305,  61.4292207 ]])
    data_I_A=[0., 1., 3., 6., 9., 7., 2., 5., 8., 4.]
    data_I_B=[2., 6., 3., 8., 1., 5., 9., 7., 4., 0.]

    # muの事前分布のパラメータを指定
    beta=1.0
    mB=np.array([60.0, 0.0, 0.0])

    # lambdaの事前分布のパラメータを指定
    w=np.array([[25**-1,0,0],[0,100**-1,0],[0,0,100**-1]])#ijime
    nu = D

    # sitaの事前分布のパラメータを指定
    alpha = np.repeat(0.1, K)
    pi=np.repeat(1/L, L)


    
    eta2B=np.zeros((N,K))
    w_hatB=np.zeros((K,D,D))

    #初期化
    #LambdaB=wishart.rvs(df=nu, scale=w, size=K)
    LambdaB=np.array(Wishart(df=nu, covariance_matrix=torch.tensor(np.tile(w,(K,1,1)))).sample())
    
    muB=MultivariateNormal(torch.tensor(mB),precision_matrix=torch.tensor(beta*LambdaB)).sample()
    sB=np.array(Multinomial(1,torch.tensor(np.array([pi]*N))).sample())
    sitaB=np.array(Dirichlet(torch.tensor(np.array([alpha]*L))).sample())
    cB=np.array(Multinomial(1,torch.tensor(sitaB[np.where(sB==1)[1]])).sample())

    #初期化(ギブス)---------------------------------------------------------
    for inter_i in range(inter_I):
        #B更新
        eta=np.dot(cB,np.log(sitaB).T)+np.log(pi)
        eta=np.exp(eta)
        eta/=np.sum(eta,axis=1,keepdims=True)
        #sB=np.array(Multinomial(1,torch.tensor(eta)).sample())

        #sB_sp = Multinomial(1,torch.tensor(eta)).sample()
        #mu,Lambdaのサンプリング-------------------------------------------------------------
        beta_hatB=np.sum(cB,axis=0)+beta
        m_hatB=(np.dot(cB.T,xB)+beta*mB)/beta_hatB.reshape(K,1)

        nu_hatB=np.sum(cB,axis=0)+nu

        for k in range(K):
            w_hatB[k]=np.dot(cB[:,k]*xB.T,xB)+beta*np.dot(mB.reshape(D, 1),mB.reshape(1, D))-beta_hatB[k]*np.dot(m_hatB[k].reshape(D, 1),m_hatB[k].reshape(1, D))+np.linalg.inv(w)
            w_hatB[k]=np.linalg.inv(w_hatB[k])
            #LambdaB[k]=wishart.rvs(size=1,df=nu_hatB[k],scale=w_hatB[k])
            LambdaB[k]=np.array(Wishart(df=nu_hatB[k], covariance_matrix=torch.tensor(w_hatB[k])).sample())
        muB=np.array(MultivariateNormal(torch.tensor(m_hatB),precision_matrix=torch.tensor(beta_hatB.reshape(K,1,1)*LambdaB)).sample())

            
        #sitaのサンプリング-------------------------------------------------------------
        alpha_hatB=np.dot(sB.T,cB)+alpha
        sitaB=np.array(Dirichlet(torch.tensor(alpha_hatB)).sample())

        for k in range(K):#cのサンプリング-------------------------------------------------------------
            eta2B[:,k]=np.diag(-0.5*np.dot(np.dot((xB-muB[k]),LambdaB[k]),(xB-muB[k]).T))
            eta2B[:,k]+=0.5*np.log(np.linalg.det(LambdaB[k]))#+1e-6

        eta2B+=np.dot(sB,np.log(sitaB))
        eta2B=np.exp(eta2B)
        eta2B/=np.sum(eta2B,axis=1,keepdims=True)
        cB=np.array(Multinomial(1,torch.tensor(eta2B)).sample())

    player.participant.LambdaB=LambdaB
    player.participant.muB=muB
    player.participant.sB=sB
    player.participant.sitaB=sitaB
    player.participant.cB=cB

    player.com_img0_cat=str(np.argmax(cB[0]))
    player.com_img1_cat=str(np.argmax(cB[1]))
    player.com_img2_cat=str(np.argmax(cB[2]))
    player.com_img3_cat=str(np.argmax(cB[3]))
    player.com_img4_cat=str(np.argmax(cB[4]))
    player.com_img5_cat=str(np.argmax(cB[5]))
    player.com_img6_cat=str(np.argmax(cB[6]))
    player.com_img7_cat=str(np.argmax(cB[7]))
    player.com_img8_cat=str(np.argmax(cB[8]))
    player.com_img9_cat=str(np.argmax(cB[9]))

    player.com_img0_sign=str(np.argmax(sB[0]))
    player.com_img1_sign=str(np.argmax(sB[1]))
    player.com_img2_sign=str(np.argmax(sB[2]))
    player.com_img3_sign=str(np.argmax(sB[3]))
    player.com_img4_sign=str(np.argmax(sB[4]))
    player.com_img5_sign=str(np.argmax(sB[5]))
    player.com_img6_sign=str(np.argmax(sB[6]))
    player.com_img7_sign=str(np.argmax(sB[7]))
    player.com_img8_sign=str(np.argmax(sB[8]))
    player.com_img9_sign=str(np.argmax(sB[9]))

    player.com_sita_lk_00=sitaB[0,0]
    player.com_sita_lk_01=sitaB[0,1]
    player.com_sita_lk_02=sitaB[0,2]
    player.com_sita_lk_10=sitaB[1,0]
    player.com_sita_lk_11=sitaB[1,1]
    player.com_sita_lk_12=sitaB[1,2]
    player.com_sita_lk_20=sitaB[2,0]
    player.com_sita_lk_21=sitaB[2,1]
    player.com_sita_lk_22=sitaB[2,2]

def com_update_and_naming(player):
    player.com_model=str(player.participant.com_model)
    #定数とハイパーパラメータ---------------------------------------------
    D=3
    K=3
    L=3
    N=10
    inter_I=100
    xB =np.array([
        [ 61.37168596,  26.15860931, -52.10096281],
        [ 68.27001472, -38.4612657 ,  53.77227298],
        [ 53.03117157, -64.87850309,  44.4914579 ],
        [ 59.5374853 , -63.855726  ,  53.98879415],
        [ 67.11591845, -54.94274545,  53.09209315],
        [ 64.58325618,  38.28864496, -51.37276997],
        [ 62.1620156 , -40.7310968 ,  51.12112248],
        [ 63.16385962,  52.90467898, -64.3546551 ],
        [ 62.34665201,  41.46976689, -22.88672083],
        [ 67.04438997, -65.46085305,  61.4292207 ]])
    data_I_A=[0., 1., 3., 6., 9., 7., 2., 5., 8., 4.]
    data_I_B=[2., 6., 3., 8., 1., 5., 9., 7., 4., 0.]

    # muの事前分布のパラメータを指定
    beta=1.0
    mB=np.array([60.0, 0.0, 0.0])

    # lambdaの事前分布のパラメータを指定
    w=np.array([[25**-1,0,0],[0,100**-1,0],[0,0,100**-1]])#ijime
    nu = D

    # sitaの事前分布のパラメータを指定
    alpha = np.repeat(0.1, K)
    pi=np.repeat(1/L, L)
    
    eta2B=np.zeros((N,K))
    w_hatB=np.zeros((K,D,D))

    #コンピュータの処理(受容確率計算，パラメータ更新)--------------------------------------------------------
    NAME={"A":0,"B":1,"C":2,"D":3,"E":4}
    NAME2=["A","B","C","D","E"]
    data_i=(player.round_number-2)%10
    
    LambdaB=player.participant.LambdaB
    muB=player.participant.muB
    sB=player.participant.sB
    sitaB=player.participant.sitaB
    cB=player.participant.cB
    
    sA_sp=NAME[player.my_sign]
    if player.com_model=="0":#MH
        r=Multinomial(1,torch.tensor(sitaB[sA_sp])).log_prob(torch.tensor(cB[int(data_I_A[data_i])])).exp()/Multinomial(1,torch.tensor(sitaB[np.argmax(sB[int(data_I_A[data_i])])])).log_prob(torch.tensor(cB[int(data_I_A[data_i])])).exp()
    elif player.com_model=="1":#all_accept
        r=1
    elif player.com_model=="2":#all_rejection
        r=0
    player.r=float(r)
    u=np.random.random(1)
    rB=r
    if torch.tensor(u)<r:#受け入れたとき
        sB[int(data_I_A[data_i])]=0
        sB[int(data_I_A[data_i])][sA_sp]=1
        player.com_accept=1
        #AC_B=[1,NAME2[sA_sp]]
    else:
        player.com_accept=0
        #AC_B=[0,NAME2[sA_sp]]
    for inter_i in range(inter_I):

        if player.com_model=="2":#all_rejection
            eta=np.dot(cB,np.log(sitaB).T)+np.log(pi)
            eta=np.exp(eta)
            eta/=np.sum(eta,axis=1,keepdims=True)
            sB=np.array(Multinomial(1,torch.tensor(eta)).sample())
        
        
        #mu,Lambdaのサンプリング-------------------------------------------------------------
        beta_hatB=np.sum(cB,axis=0)+beta
        m_hatB=(np.dot(cB.T,xB)+beta*mB)/beta_hatB.reshape(K,1)
        nu_hatB=np.sum(cB,axis=0)+nu

        for k in range(K):
            w_hatB[k]=np.dot(cB[:,k]*xB.T,xB)+beta*np.dot(mB.reshape(D, 1),mB.reshape(1, D))-beta_hatB[k]*np.dot(m_hatB[k].reshape(D, 1),m_hatB[k].reshape(1, D))+np.linalg.inv(w)
            w_hatB[k]=np.linalg.inv(w_hatB[k])
            #LambdaB[k]=wishart.rvs(size=1,df=nu_hatB[k],scale=w_hatB[k])
            LambdaB[k]=np.array(Wishart(df=nu_hatB[k], covariance_matrix=torch.tensor(w_hatB[k])).sample())
        muB=np.array(MultivariateNormal(torch.tensor(m_hatB),precision_matrix=torch.tensor(beta_hatB.reshape(K,1,1)*LambdaB)).sample())

            
        #sitaのサンプリング-------------------------------------------------------------
        alpha_hatB=np.dot(sB.T,cB)+alpha
        sitaB=np.array(Dirichlet(torch.tensor(alpha_hatB)).sample())

        for k in range(K):#cのサンプリング-------------------------------------------------------------
            eta2B[:,k]=np.diag(-0.5*np.dot(np.dot((xB-muB[k]),LambdaB[k]),(xB-muB[k]).T))
            eta2B[:,k]+=0.5*np.log(np.linalg.det(LambdaB[k]))#+1e-6

        eta2B+=np.dot(sB,np.log(sitaB))
        eta2B=np.exp(eta2B)
        eta2B/=np.sum(eta2B,axis=1,keepdims=True)

        cB=np.array(Multinomial(1,torch.tensor(eta2B)).sample())


    #コンピュータの名付け(サインのサンプリング)------------------------------------------------
    eta=np.dot(cB,np.log(sitaB).T)+np.log(pi)
    eta=np.exp(eta)
    eta/=np.sum(eta,axis=1,keepdims=True)
    #s=np.array(Multinomial(1,torch.tensor(eta)).sample())        
    sB_sp = Multinomial(1,torch.tensor(eta[int(data_I_B[data_i])])).sample()
    sB_sp2=np.argmax(sB_sp)

    #signB=NAME2[sB_sp2]#名付けを受信
    player.com_sign=NAME2[sB_sp2]#名付けを受信
    

    player.participant.LambdaB=LambdaB
    player.participant.muB=muB
    player.participant.sB=sB
    player.participant.sitaB=sitaB
    player.participant.cB=cB

    player.com_img0_cat=str(np.argmax(cB[0]))
    player.com_img1_cat=str(np.argmax(cB[1]))
    player.com_img2_cat=str(np.argmax(cB[2]))
    player.com_img3_cat=str(np.argmax(cB[3]))
    player.com_img4_cat=str(np.argmax(cB[4]))
    player.com_img5_cat=str(np.argmax(cB[5]))
    player.com_img6_cat=str(np.argmax(cB[6]))
    player.com_img7_cat=str(np.argmax(cB[7]))
    player.com_img8_cat=str(np.argmax(cB[8]))
    player.com_img9_cat=str(np.argmax(cB[9]))

    player.com_img0_sign=str(np.argmax(sB[0]))
    player.com_img1_sign=str(np.argmax(sB[1]))
    player.com_img2_sign=str(np.argmax(sB[2]))
    player.com_img3_sign=str(np.argmax(sB[3]))
    player.com_img4_sign=str(np.argmax(sB[4]))
    player.com_img5_sign=str(np.argmax(sB[5]))
    player.com_img6_sign=str(np.argmax(sB[6]))
    player.com_img7_sign=str(np.argmax(sB[7]))
    player.com_img8_sign=str(np.argmax(sB[8]))
    player.com_img9_sign=str(np.argmax(sB[9]))

    player.com_sita_lk_00=sitaB[0,0]
    player.com_sita_lk_01=sitaB[0,1]
    player.com_sita_lk_02=sitaB[0,2]
    player.com_sita_lk_10=sitaB[1,0]
    player.com_sita_lk_11=sitaB[1,1]
    player.com_sita_lk_12=sitaB[1,2]
    player.com_sita_lk_20=sitaB[2,0]
    player.com_sita_lk_21=sitaB[2,1]
    player.com_sita_lk_22=sitaB[2,2]

# PAGES
class Categorization(Page):
    #init_com_paramater(Player)
            
    form_model = Player
    form_fields = ["img0_sign","img1_sign","img2_sign","img3_sign","img4_sign","img5_sign","img6_sign","img7_sign","img8_sign","img9_sign",
                   "order_A0","order_A1","order_A2","order_A3","order_A4","order_A5","order_A6","order_A7","order_A8","order_A9",
                   "order_B0","order_B1","order_B2","order_B3","order_B4","order_B5","order_B6","order_B7","order_B8","order_B9",
                   "order_C0","order_C1","order_C2","order_C3","order_C4","order_C5","order_C6","order_C7","order_C8","order_C9",]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        if player.round_number == 1:
            init_com_paramater(player)
            


class MyPage_test(Page):
    form_model = Player
    
class Naming(Page):
    form_model = Player
    form_fields = ["my_sign","img0_sign","img1_sign","img2_sign","img3_sign","img4_sign","img5_sign","img6_sign","img7_sign","img8_sign","img9_sign",
                   "order_A0","order_A1","order_A2","order_A3","order_A4","order_A5","order_A6","order_A7","order_A8","order_A9",
                   "order_B0","order_B1","order_B2","order_B3","order_B4","order_B5","order_B6","order_B7","order_B8","order_B9",
                   "order_C0","order_C1","order_C2","order_C3","order_C4","order_C5","order_C6","order_C7","order_C8","order_C9"]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1
    
    @staticmethod
    def js_vars(player: Player):
        prev_player = player.in_round(player.round_number - 1)
        prev_order_list=[[prev_player.order_A0,prev_player.order_A1,prev_player.order_A2,prev_player.order_A3,prev_player.order_A4,prev_player.order_A5,prev_player.order_A6,prev_player.order_A7,prev_player.order_A8,prev_player.order_A9],
                         [prev_player.order_B0,prev_player.order_B1,prev_player.order_B2,prev_player.order_B3,prev_player.order_B4,prev_player.order_B5,prev_player.order_B6,prev_player.order_B7,prev_player.order_B8,prev_player.order_B9],
                        [prev_player.order_C0,prev_player.order_C1,prev_player.order_C2,prev_player.order_C3,prev_player.order_C4,prev_player.order_C5,prev_player.order_C6,prev_player.order_C7,prev_player.order_C8,prev_player.order_C9],]
        data_I_A=[0., 1., 3., 6., 9., 7., 2., 5., 8., 4.]
        return {"prev_order_list":prev_order_list,"data_i":data_I_A[(player.round_number-2)%10],"progr":player.round_number}

class Accept_or_Rejection(Page):
    form_model = Player
    form_fields = ["my_accept","img0_sign","img1_sign","img2_sign","img3_sign","img4_sign","img5_sign","img6_sign","img7_sign","img8_sign","img9_sign",
                   "order_A0","order_A1","order_A2","order_A3","order_A4","order_A5","order_A6","order_A7","order_A8","order_A9",
                   "order_B0","order_B1","order_B2","order_B3","order_B4","order_B5","order_B6","order_B7","order_B8","order_B9",
                   "order_C0","order_C1","order_C2","order_C3","order_C4","order_C5","order_C6","order_C7","order_C8","order_C9"]
    
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1

    @staticmethod
    def vars_for_template(player: Player):
        if player.round_number > 1:
            com_update_and_naming(player)
    
    @staticmethod
    def js_vars(player: Player):
        prev_player = player
        prev_order_list=[[prev_player.order_A0,prev_player.order_A1,prev_player.order_A2,prev_player.order_A3,prev_player.order_A4,prev_player.order_A5,prev_player.order_A6,prev_player.order_A7,prev_player.order_A8,prev_player.order_A9],
                         [prev_player.order_B0,prev_player.order_B1,prev_player.order_B2,prev_player.order_B3,prev_player.order_B4,prev_player.order_B5,prev_player.order_B6,prev_player.order_B7,prev_player.order_B8,prev_player.order_B9],
                        [prev_player.order_C0,prev_player.order_C1,prev_player.order_C2,prev_player.order_C3,prev_player.order_C4,prev_player.order_C5,prev_player.order_C6,prev_player.order_C7,prev_player.order_C8,prev_player.order_C9],]
        data_I_B=[2., 6., 3., 8., 1., 5., 9., 7., 4., 0.]
        return {"prev_order_list":prev_order_list,"data_i":data_I_B[(player.round_number-2)%10],"progr":player.round_number}

class Accept_or_Rejection2(Page):
    form_model = Player
    form_fields = ["img0_sign","img1_sign","img2_sign","img3_sign","img4_sign","img5_sign","img6_sign","img7_sign","img8_sign","img9_sign",
                   "order_A0","order_A1","order_A2","order_A3","order_A4","order_A5","order_A6","order_A7","order_A8","order_A9",
                   "order_B0","order_B1","order_B2","order_B3","order_B4","order_B5","order_B6","order_B7","order_B8","order_B9",
                   "order_C0","order_C1","order_C2","order_C3","order_C4","order_C5","order_C6","order_C7","order_C8","order_C9"]
    
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1
    @staticmethod
    def js_vars(player: Player):
        prev_player = player
        prev_order_list=[[prev_player.order_A0,prev_player.order_A1,prev_player.order_A2,prev_player.order_A3,prev_player.order_A4,prev_player.order_A5,prev_player.order_A6,prev_player.order_A7,prev_player.order_A8,prev_player.order_A9],
                         [prev_player.order_B0,prev_player.order_B1,prev_player.order_B2,prev_player.order_B3,prev_player.order_B4,prev_player.order_B5,prev_player.order_B6,prev_player.order_B7,prev_player.order_B8,prev_player.order_B9],
                        [prev_player.order_C0,prev_player.order_C1,prev_player.order_C2,prev_player.order_C3,prev_player.order_C4,prev_player.order_C5,prev_player.order_C6,prev_player.order_C7,prev_player.order_C8,prev_player.order_C9],]
        data_I_B=[2., 6., 3., 8., 1., 5., 9., 7., 4., 0.]
        return {"prev_order_list":prev_order_list,"data_i":data_I_B[(player.round_number-2)%10],"progr":player.round_number}




page_sequence =[Categorization ,Naming, Accept_or_Rejection,Accept_or_Rejection2]
