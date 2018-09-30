import re


def jia(*args):
    '''
    加法解析
    :param args:
    :return:
    '''
    res = 0
    for i in args:
        #解决*-和/-的情况
        #i = i.replace('*-','#').replace('/-','@')

        jian_list = i.split('-')
        # 减号的拆分要解决负号的情况:把负号拆分后列表中所有为空字符串的元素替换成'0'
        # 还要把*-和/-替换回来
        for id,j in enumerate(jian_list):
            if j=='':
                jian_list[id] = '0'
            else:
                jian_list[id] = j.replace('#','*-').replace('@','/-')

        res += jian(*jian_list)
    return res
def jian(*args):
    '''
    减法解析
    :param args:
    :return:
    '''
    #减法需要先算出第一个式子，拿第一个式子去减后面的式子
    res = args[0:1]

    res = chen(*res[0].split('*'))

    for i in args[1:]:
        chen_list = i.split('*')
        res -= chen(*chen_list)
    return res
def chen(*args):
    '''
    乘法解析
    :param args:
    :return:
    '''
    res = 1
    for i in args:
        chu_list = i.split('/')
        res *= chu(*chu_list)
    return res
def chu(*args):
    '''
    除法解析
    :param args:
    :return:
    '''
    #除法需要先取出第一个，拿第一个除以后面的
    res = args[0:1]

    res = float(res[0])

    for i in args[1:]:
        i = float(i)
        res /= i
    return res

def simpleCalc(input_str):
    '''
    不带括号的表达式解析
    :param input_str:
    :return:
    '''
    # 去掉最外层的括号
    input_str = input_str.strip('()')
    # 处理 --、+- 的情况，还有 *-、/- 的情况没处理
    input_str = input_str.replace(' ','').replace('--','+').replace('+-','-').replace('*-','#').replace('/-','@')
    #print(input_str)

    #计算加减乘除
    jia_list = input_str.split('+')
    res = jia(*jia_list)
    return res
    #return str(eval(input_str))

def calc(input_str):
    '''
    计算器入口
    :param input_str:
    :return:
    '''
    if len(input_str) == 0:
        print('Wrong input')
        exit(0)
    #print(input_str)
    #查找是否还有括号
    m = re.search('\([^()]+\)', input_str)
    brackets_exists = False
    #print(m)
    if m == None:#不再有括号了，就直接计算
        simple_calc_str = input_str#需要计算的值的字符串就是传入的表达式
    else:#还有括号，就把找到的括号中的表达式计算的值替换括号所在位置
        brackets_exists = True
        simple_calc_str = m.group()#需要计算值的字符串是找到的括号中的表达式

    simple_res = str(simpleCalc(simple_calc_str))
    if brackets_exists:#还有括号，就把找到的括号中的表达式计算的值替换括号所在位置,进入迭代
        return calc(input_str.replace(simple_calc_str,simple_res,1))
    else:#没有括号就直接把计算结果返回
        return simple_res

if __name__ == '__main__':
    # input_str = '3 * 4 + (-4 / 2 - 8 - 3 * 2 + ( 4 - 5 / 2 + 11 - ( 2 * 3 - 9 ) - 12 )) + 20 - 3 * 2 - ( 5 + 8 / 4)'
    # input_str = '3/(-1) - (4*-2)/(1+1)/(1+1)'
    input_str = '1-2*((60-30+(-40/5)*(9-2*5/3+7/3*99/4*2998+10*568/14))-(-4*3)/(16-3*2))'
    input_str = '2 + 2 * 3'

    result = calc(input_str)
    print(result)
