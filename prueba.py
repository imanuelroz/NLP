def isIsomorphic(s, t):
    s1 = set(s)
    t1 = set(t)
    if len(s) != len(t):
        return False
    if len(s1) != len(t1):
        return False
    if len(s1) != len(s) and len(t1) != len(t):
        s_v = []
        t_v = []
        for i in range(len(s)):
            s_v.append(s[i])
            t_v.append(t[i])
        for i in range(len(s)):
            t_v[i] = s_v[i]
            print(t_v)
            #print(s)
        if t_v != s_v:
            print('Not isomorphic')
            return False
        else:
            print('They are isomorphic')
            return True
    else:
        return True
isIsomorphic("bbbaaaba", "aaabbbba")
a, b = "bbbaaaba", "aaabbbba"



x = [1,2,3,4]
y = [2,1,1,5]
z_x = 0
z_y = 0
for i in range(len(y)):
    z_x += abs(x[i])
    z_y += abs(y[i])
print(max(z_x, z_y))


#TOXIC COMMENT CLASSIFICATION: MULTI-LABEL text Classification problem with highly imbalanced dataset
#multiple types of tosicity like: threats, obscenity, insults, and identity-based hate.


