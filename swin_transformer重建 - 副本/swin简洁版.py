# _*_coding:utf-8_*_
###
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

### 如何基于图片生成 patch embedding ?

'''
铺垫:
方法一：
    基于pytorch unfold的API来将图片进行分块儿， 也就是模仿卷积的思路， 设置kernel_size=stride=patch_size, 得到分块儿后的图片
    得到格式为[bs, num_patch, patch_depth]的张量
    和张量和形状为[patch_depth, model_dim_C]的权重矩阵进行乘法操作， 即可得到形状为[bs, num_patch, model_dim_C]的patch_embadding
    
方法二：
    patch_depth是等于input_channel*patch_size
    model_dim_C相当于二维卷积的输出通道数目
    将形状为[patch_depth, model_dim_C, input_channel, patch_size, patch_size]的卷积核
    调用Pytorch中的conv3d API得到卷积的输出张量， 形状为[bs, output_channel, height, width]
    转换为[bs, num_patch, model_dim_C]的格式，即为patch embadding
    num_patch = patch_size * patch_size 
'''
# 难点1 patch embedding
def image2emb_naive(image, patch_size,weight):
    """naive 直观方法实现patch embadding"""
    # image shape : bs*channel*h*w (pytorch要求的标准图片输入格式)
    patch = F.unfold(input=image,kernel_size=(patch_size,patch_size),stride=(patch_size,patch_size)).transpose(-1,-2) # patch [bs,num_patch,patch_depth]
    patch_embadding = patch @ weight
    return patch_embadding

def image2emb_conv(image, kernel ,stride):
    """ 基于二维卷积来实现patch embadding ,embedding的维度就是卷积的输出通道数"""
    # bs*oc*oh*ow
    conv_output = F.conv2d(input=image,weight=kernel,stride=stride)
    bs , oc ,oh ,ow = conv_output.shape
    patch_embedding = conv_output.reshape((bs, oc, oh*ow)).transpose(-1 , -2)
    return patch_embedding



### 如何构建MHSA并计算其复杂度？
'''
铺垫：
基于输入x进行三个映射分别得到q,k,v
    此步的复杂度为3LC(2), 其中L为序列长度， C为特征大小
将q,k,v拆分成多头的形式，注意这里的多头各自计算，互不影响， 所以可以与bs维度进行统一的看待
计算q乘以k的转置， 并考虑可能的掩码， 即让无效的两两位置之间的能量为负无穷， 掩码是在shift window MHSA中会需要， 而在window MHSA中暂不需要
    此步的计算复杂度为L(2)C
计算概率值与v的乘积
    此步的计算复杂度为L(2)C
对输出进行再次映射
    此步的计算复杂度为LC(2)
总体复杂度为 4LC(2) + 2L(2)C

'''
# 为什么要创建成类呢？
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, model_dim, num_head):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_head = num_head

        self.proj_linear_layer = nn.Linear(in_features=model_dim, out_features=3*model_dim)
        self.final_linear_layer = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self,input,additive_mask=None):
        bs ,seqlen ,model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head # // 代表啥意思?

        proj_output = self.proj_linear_layer(input)
        q, k ,v = proj_output.chunk(3,dim=-1) # [bs, seqlen, model_dim]

        q = q.reshape(bs, seqlen, num_head, head_dim).transpose(1,2)  # [bs, num_head, seqlen, head_dim]
        q = q.reshape(bs*num_head,seqlen,head_dim)

        k = k.reshape(bs, seqlen, num_head,head_dim).transpose(1,2) # [bs, num_head, seqlen, head_dim]
        k = k.reshape(bs*num_head,seqlen,head_dim)

        v = v.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2)  # [bs, num_head, seqlen, head_dim]
        v = v.reshape(bs * num_head, seqlen, head_dim)

        if additive_mask is None:
            attn_prob = F.softmax(torch.bmm(q,k.transpose(-2,-1))/math.sqrt(head_dim),dim=-1)
        else:
            additive_mask = additive_mask.tile((num_head,1,1))
            attn_prob = F.softmax(torch.bmm(q,k.transpose(-2,-1))/math.sqrt(head_dim),dim=-1)

        #做注意力机制，算出输出
        output = torch.bmm(attn_prob,v) # [bs*num_head, seqlen, head_dim]
        output = output.reshape(bs, num_head, seqlen, head_dim).transpose(1,2) # [bs, seq_len, num_head, head_dim]
        output = output.reshape(bs, seqlen, model_dim)

        output = self.final_linear_layer(output)
        return attn_prob, output

### 3 如何构建 window MHSA 并计算其复杂度？
'''
铺垫：
将patch组成的图片进一步划分成一个个更大的window
    首先需要将三维的patch embadding转换成图片的格式
    使用unfold来将patch划分成window
在每个window内部计算MHSA
    window数目其实可以跟batch_size进行统一的对待，因为window与window之间没有交互计算
    关于计算复杂度
        假设窗的边长为W， 那么计算每个窗的总体复杂度是 4W(2)C(2) + 2W(4)C
        假设patch的总数目为L， 那么窗的数目就为L/W(2)
        因此，W-MHSA的总体复杂度为4LC(2) + 2LW(2)C
    此处不需要mask
    将计算结果转换成带window的四维张量格式
    复杂度对比
        MHSA： 4LC(2) + 2L(2)C
        W-MHSA: 4LC(2) + 2LW(2)C
'''
# W 固定的 所以W(2)可以看成一个常数 由L(2)变为了LW(2) 所以就大致可以看作是由平方复杂度 变成了 线性复杂度

def window_multi_head_self_attention(patch_embadding, mhsa, window_size=4, num_head=2):
    num_patch_in_window = window_size * window_size
    bs, num_patch ,patch_depth = patch_embadding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    patch_embadding = patch_embadding.transpose(-1, -2)
    patch = patch_embadding.reshape(bs, patch_depth, image_height, image_width)
    window = F.unfold(input=patch, kernel_size=(window_size,window_size), stride=(window_size,window_size)).transpose(-1,-2)

    bs, num_window, patch_depth_times_num_patch_in_window = window.shape
    # [bs*num_window,num_patch,patch_depth]
    window = window.reshape(bs*num_window, patch_depth, num_patch_in_window).transpose(-1,-2)

    attn_prob , output = mhsa(window)       # [bs*num_window ,num_patch_in_window, patch_depth]

    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    return output



###4 如何构建shift window MHSA及其mask?
'''
铺垫：
将上一步的w-mhsa的结果转换成图片的格式
假设已经做了新的window划分，这一步叫做shift-window
为了保持window的数目的不变从而有高效的计算， 需要将图片的patch往左和往上各自滑动半个窗口的步长， 保持patch所属window类别不变
将图片patch还原成window的数据格式
由于shift-window后，每个window虽然形状规整，但部分window中存在原本不属于同一个窗口的patch, 所以需要生成mask
如何生成mask?
    首先构建一个shift-window的patch所属的window类别矩阵
    对该矩阵进行同样的往左和往上各自滑动半个窗口大小的步长的操作
    通过unfold操作得到【bs, num_window, num_patch_in_window】形状的类别矩阵
    对该矩阵进行扩维度成【bs, num_window, num_patch_in_window, 1】
    将该矩阵与其转置矩阵进行作差，得到同类关系矩阵（为0的位置上的patch属于同类，否则属于不同类）
    对同类关系矩阵中非零的位置用负无穷进行填充 ，使之能量变为负无穷，对于零的位置用0去填充，这样就 构建好了MHSA所需要的mask
    此mask的形状为【bs*num_window, num_patch_in_window,num_patch_in_window】
将window转换成三维的格式，【bs*num_window, num_patch_in_window, patch_depth】
将三维格式的特征连同mask一起送入MHSA中计算得到注意力输出
将注意力输出转换成图片的patch格式，【bs, num_window, num_patch_in_window, patch_depth】
为了恢复位置， 需要将图片的patch往右和往下各自滑动半个窗口大小的步长， 至此， SW-MHSA计算完毕

'''

# 定义一个辅助函数， window2iamge, 也就是将transformer block的结果转化成图片的格式
def window2iamge(msa_output):
    bs, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_height = int(math.sqrt(num_window)) * window_size
    image_width = image_height

    msa_output = msa_output.reshape(bs,
                                    int(math.sqrt(num_window)),
                                    int(math.sqrt(num_window)),
                                    window_size,
                                    window_size,
                                    patch_depth)
    msa_output = msa_output.transpose(2,3)
    image = msa_output.reshape(bs, image_height*image_width, patch_depth)

    image = image.transpose(-1, -2).reshape(bs, patch_depth, image_height, image_width) #跟卷积的格式一致

    return image

# 定义辅助函数 shift_window , 即高效地计算swmsa
def shift_window(w_msa_output, window_size, shift_size, generate_mask=False):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    w_msa_output = window2iamge(msa_output=w_msa_output) # 【bs, depth, h, w】
    bs, patch_depth, image_height, image_width = w_msa_output.shape

    rolled_w_msa_output = torch.roll(input=w_msa_output, shifts=(shift_size,shift_size), dims=(2,3))

    shifted_w_msa_input = rolled_w_msa_output.reshape(bs,
                                                      patch_depth,
                                                      int(math.sqrt(num_window)),
                                                      window_size,
                                                      int(math.sqrt(num_window)),
                                                      window_size)

    shifted_w_msa_input = shifted_w_msa_input.transpose(3,4)
    shifted_w_msa_input = shifted_w_msa_input.reshape(bs, patch_depth, num_window*num_patch_in_window)
    shifted_w_msa_input = shifted_w_msa_input.transpose(-1,-2) # [bs,num_window*num_patch_in_window, patch_depth]
    shifted_window = shifted_w_msa_input.reshape(bs, num_window, num_patch_in_window, patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(bs, image_height, image_width, window_size)
    else:
        additive_mask = None

    return shifted_window,additive_mask
# 构建shift window multi-head attention mask
def build_mask_for_shifted_wmsa(batch_size, image_height, image_width, window_size):
    index_metrix = torch.zeros(image_height,image_width)

    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2) // window_size
            col_times = (i+window_size//2) // window_size
            index_metrix[i,j] = row_times*(image_height//window_size) + col_times + 1
    rolled_index_metrix = torch.roll(input=index_metrix,shifts=(-window_size//2,-window_size//2),dims=(0,1))
    rolled_index_metrix = rolled_index_metrix.unsqueeze(0).unsqueeze(0)   #[bs, ch, h, w]

    c = F.unfold(input=rolled_index_metrix, kernel_size=(window_size,window_size),stride=(window_size,window_size)).transpose(-1,-2)

    c = c.tile(batch_size, 1, 1)   #[batch_size, num_windows, num_patch_in_window]

    bs, num_window, num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1) #[batch_size, num_window, num_patch_in_window, num_patch_in_window]
    c2 = (c1 - c1.transpose(-1,-2)) == 0 #[bs, num_window, num_patch_in_window, num_patch_in_window]
    valid_matrix = c2.to(torch.float32)
    additive_mask = (1-valid_matrix)*(-1e-9) # [batch_size, num_window, num_patch_in_window, num_patch_in_window]

    additive_mask = additive_mask.reshape(bs*num_window, num_patch_in_window, num_patch_in_window)

    return additive_mask

def shift_window_multi_head_self_attention(w_msa_output, mhsa, window_size=4, num_head=2):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    shifted_w_msa_input, additive_mask = shift_window(w_msa_output=w_msa_output,window_size=window_size,shift_size=window_size//2,
                                                      generate_mask=True)

    print(shifted_w_msa_input.shape) #【batch_size,num_window,num_patch_in_window, patch_depth】
    print(additive_mask.shape)   # [batch_size * num_window, num_patch_in_window, num_patch_in_window]

    shifted_w_msa_input = shifted_w_msa_input.reshape(bs*num_window, num_patch_in_window, patch_depth)

    attn_prob, output = mhsa(shifted_w_msa_input,additive_mask=additive_mask)

    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)

    output,_ = shift_window(w_msa_output=output, window_size=window_size, shift_size=window_size//2, generate_mask=False)
    print(output.shape) # [batch_size, num_window, num_patch_in_window, patch_depth]
    return output




### 5 如何构建patch merging？
'''
铺垫：
将window格式的特征转换成图片patch格式
利用unfold操作，按照merge_size*merge_size的大小得到新的patch, 形状为【bs, num_patch_new, merge_size*merge_size*patch_depth_old】
使用一个全连接层对depth进行降维成0.5倍， 也就是从merge_size*merge_size*patch_depth_old 映射到 0.5*merge_size*merge_size*patch_depth_old
输出的是patch embedding的形状格式 ，【bs, num_patch, patch_depth】
举例说明： 以merge_size=2为例， 经过patchmerging后， patch数目减少为之前的1、4， 但是patch增大为原来的2倍， 而不是4倍'''
# 难点4 构建 patch merging
class PatchMerging(nn.Module):

    def __init__(self,model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(in_features=model_dim*merge_size*merge_size,
                                    out_features= int(model_dim * merge_size * merge_size * output_depth_scale))

    def forward(self,input):
        bs, num_window, num_patch_in_window, patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))

        input = window2iamge(msa_output=input) #[bs, patch_depth, image_height, image_window]

        merged_window = F.unfold(input=input,kernel_size=(self.merge_size,self.merge_size),stride=(self.merge_size,self.merge_size)).transpose(-1,-2)


        merged_window = self.proj_layer(merged_window)

        return merged_window




### 6 如何构建swintransformerblock
'''
铺垫：
每个block包含 layernorm W-MHSA MLP SW-MHSA，残差连接等模块儿
输入是patch embedding格式 
每个mlp包含两层， 分别是4*model_dim 和 model_dim 的大小
输出的是window的数据格式， 【bs, num_window, num_patch_in_window, patch_depth】
需要注意的是 残差连接时对数据形状的要求
'''
class SwintransformerBlock(nn.Module):

    def __init__(self,model_dim, window_size, num_head):
        super(SwintransformerBlock, self).__init__()
        self.layer_noem1 = nn.LayerNorm(model_dim)
        self.layer_noem2 = nn.LayerNorm(model_dim)
        self.layer_noem3 = nn.LayerNorm(model_dim)
        self.layer_noem4 = nn.LayerNorm(model_dim)

        self.wsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.wsma_mlp2 = nn.Linear(4*model_dim,model_dim)
        self.swsma_mlp1 = nn.Linear(model_dim, 4*model_dim)
        self.swsma_mlp2 = nn.Linear(4*model_dim, model_dim)

        self.mhsa1 = MultiHeadSelfAttention(model_dim, num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim, num_head)

    def forward(self, input):

        bs, num_patch, patch_depth = input.shape

        input1 = self.layer_noem1(input)
        w_msa_output = window_multi_head_self_attention(patch_embadding=input,mhsa=self.mhsa1,window_size=4,num_head=2)
        bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
        w_msa_output = input + w_msa_output.reshape(bs, num_patch, patch_depth)
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_noem2(w_msa_output)))
        output1 += w_msa_output

        input2 = self.layer_noem3(output1)
        input2 = input2.reshape(bs, num_window, num_patch_in_window,patch_depth)
        sw_msa_output = shift_window_multi_head_self_attention(w_msa_output=input2,mhsa=self.mhsa2,window_size=4,num_head=2)
        sw_msa_output = output1 + sw_msa_output.reshape(bs, num_patch, patch_depth)
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_noem4(sw_msa_output)))
        output2 += sw_msa_output

        output2 = output2.reshape(bs, num_window, num_patch_in_window, patch_depth)

        return output2




### 7 如何构建swintransformermodel？
'''
铺垫：
输入的是图片
首先对图片进行分块并得到patch embedding
经过第一个stage
进行patch merging ，再进行第二个stage
以此类推。。。
对最后一个block的输出转换成patch embedding 的格式 【bs， num_patch, patch_depth】
对patch embedding在时间维度进行平均池化， 并映射到分类层得到分类的logits， 完毕

'''
class SwinTransformerModel(nn.Module):
    def __init__(self,input_image_channel=3, patch_size=4, model_dim_C=8, num_classes=10, window_size=4, num_head=2, merge_size=2):

        super(SwinTransformerModel, self).__init__()
        patch_depth = patch_size*patch_size*input_image_channel
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_C))

        self.block1 = SwintransformerBlock(model_dim_C, window_size, num_head)
        self.block2 = SwintransformerBlock(model_dim_C*2, window_size, num_head)
        self.block3 = SwintransformerBlock(model_dim_C*4, window_size, num_head)
        self.block4 = SwintransformerBlock(model_dim_C*8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_C, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C*2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C*4, merge_size)

        self.final_layer = nn.Linear(in_features=model_dim_C*8, out_features=num_classes)

    def forward(self,image):
        patch_embedding_naive = image2emb_naive(image, self.patch_size, self.patch_embedding_weight)
        # print(patch_embedding_naive)
        #
        # kernel = self.patch_embedding_weight.transpose(0,1).reshape((-1,ic,patch_size,patch_size)) #oc*ic*kh*kw
        # patch_embedding_conv = image2emb_conv(image,kernel=kernel,stride=self.patch_size)   # 二维卷积地方法得到embedding
        # print(patch_embedding_conv)

        #block1
        patch_embedding = patch_embedding_naive
        print(patch_embedding.shape,'1')

        sw_msa_output = self.block1(patch_embedding)
        print("block1_output",sw_msa_output.shape) #[bs, num_window, num_patch_in_window, patch_depth]

        merged_patch1 = self.patch_merging1(sw_msa_output)
        print(merged_patch1.shape,'2')
        sw_msa_output_1 = self.block2(merged_patch1)
        print("block2_output", sw_msa_output_1.shape)

        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)
        print("block3_output",sw_msa_output_2.shape)

        merged_patch3 = self.patch_merging3(sw_msa_output_2)
        sw_msa_output_3 = self.block4(merged_patch3)
        print("block4_output",sw_msa_output_3.shape)

        bs, num_window, num_patch_in_window, patch_depth = sw_msa_output_3.shape
        sw_msa_output_3 = sw_msa_output_3.reshape(bs, -1, patch_depth)

        pool_output = torch.mean(sw_msa_output_3, dim=1)
        logits = self.final_layer(pool_output)
        print("logits",logits.shape)

        return logits
### 8 模型测试代码
# 难点5： 分类模块儿

if __name__=="__main__":
    bs, ic, image_h, image_w = 4, 3, 256, 256
    patch_size = 4
    model_dim_C = 8 #一开始的patch embedding的大小
    # max_num_torkn = 16
    num_classes = 10
    window_size = 4
    num_head = 2
    merge_size = 2

    patch_depth = patch_size*patch_size*ic
    image = torch.randn(bs,ic,image_h,image_w)
    print(image.shape)

    model = SwinTransformerModel(ic,patch_size,model_dim_C,num_classes,window_size,num_head,merge_size)

    logits = model(image)
    print(logits)