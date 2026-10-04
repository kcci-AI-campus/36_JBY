# N = int(input())

# # 코드 작성
# R = N//5


# print(R)






# S1, S2 = input().split()

# S1 = int(S1)
# S2 = int(S2)
# print(S1, S2)
# # 코드 작성
# R = max(S1, S2)

# print(R)








# x = eval(input())

# # 코드 작성
# R = x[-1]

# print(R)










# x = eval(input())

# # 코드 작성

# l1 = x[::2]
# l2 = x[1::2]
# m1 = min(l1)
# m2 = min(l2)

# print(m1, m2)




# x = list(map(float(input().split())))


# x = eval(input())

# # 코드 작성
# result1=0
# for i in range(len(x)):
#     result1 += x[i]

# R = result1/len(x)

# print(R)





# x = eval(input())

# # 코드 작성

# # print(x)

# y = sorted(x, key=lambda x: (x[1], x[0]))

# print(y)












# kor = eval(input())
# eng = eval(input()) 

# # 코드 작성
# #print(kor)
# #print(eng)

# #print(kor.index(max(kor)))

# R = eng[kor.index(max(kor))]


# print(R)





# N = int(input())

# # 코드 작성

# if N<5:
#     R = N
# elif N<8:
#     R = N + 1
# else:
#     R = N + 2


# print(R)










# x = eval(input())
# y=[]
# # 코드 작성
# for i in range(len(x)):
#     #print(x[i])
#     if (len(x[i]) >= 3) : 
#         y.append(x[i][::-1])
        
# print(y)








# t = eval(input())
# h = float(input())

# # 코드 구현

# #print(t)

# r=0

# for i in range(len(t)):
#     if t[i] > h:
#         r=r+1



# print(r)







t = eval(input())
h = float(input())

# 코드 구현

#print(t)

r=0

def pp(x):
    return x+1

r = (pp(i) for i in range(len(t)) if t[i] > h)
    # if t[i] > h:
    #     r=r+1
print(r)
r = list(r)



print(r)














