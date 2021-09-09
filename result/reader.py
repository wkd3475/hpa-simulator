import csv
import matplotlib.pyplot as plt

filename = "test3/3-action-q-learning/q_leanring_fixed_action_1_t19_41_00"

bad_file = open(f'./{filename}.csv', 'r', encoding='utf-8')
rdr =csv.reader(bad_file)

result_history = None
result = None
error_rate = None
avg_pods = None
avg_cost = None
avg_util = None
it = None
A = None
E = None
C = None

j = 0
for line in rdr:
    if j==0:
        result_history = line
    
    if j==1:
        result = line

    if j==2:
        it = line

    if j==3:
        A = line

    if j==4:
        E = line

    if j==5:
        C = line

    j = j+1

error_rate = result[0]
avg_pods = result[1]
avg_cost = result[2]
avg_util = result[3]

for i in range(len(result_history)):
    result_history[i] = float(result_history[i])

for i in range(len(A)):
    it[i] = int(it[i])
    A[i] = float(A[i])
    E[i] = float(E[i])
    C[i] = int(C[i])


print(filename)
print(f"error rate : {error_rate}")
print(f"avg_pods : {avg_pods}")
print(f"avg_cost : {avg_cost}")
print(f"avg_util : {avg_util}")

x = list(range(len(result_history)))
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x, result_history, 'r', label='total rewards')
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.3)
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('Autoscale Period')
ax1.set_ylabel('Number of Requests')
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Autoscale Period')
ax2.set_ylabel('Number of Pods')

ax1.plot(it, A, 'r', label='A')
ax1.plot(it, E, 'b', label='E')
ax2.plot(it, C, 'g', label='C')
plt.show()