from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

data = np.loadtxt("new-data.txt", skiprows=1, delimiter=",")
T,Q= data.T



# popt, pcov = curve_fit( T, Q)

def decline_curve(curve_type, q_i):
    if curve_type == "exponential":
        def exponential_decline(T, a):
            # print(T, a)
            return q_i*np.exp(-a*T)
        return exponential_decline
    
    elif curve_type == "hyperbolic":
        def hyperbolic_decline(T, a_i, b):
            # print("ai;", a_i, "\n", 'b:', b)
            return q_i/np.power((1+b*np.abs(a_i)*T), 1/b)
            #return q_i/((1+b*a_i*T)**(1/b))
        return hyperbolic_decline
    
    elif curve_type == "harmonic":
        def parabolic_decline(T, a_i):
            return q_i/(1+a_i*T)
        return parabolic_decline
    
    else:
        raise "I don't know this decline curve!"

def L2_norm(Q, Q_obs):
    return np.sum(np.power(np.subtract(Q, Q_obs), 2))



exp_decline = decline_curve("exponential", Q[0])
hyp_decline = decline_curve("hyperbolic", Q[0])
har_decline = decline_curve("harmonic", Q[0])

popt_exp, pcov_exp = curve_fit(exp_decline, T, Q, method="trf")
popt_hyp, pcov_hyp = curve_fit(hyp_decline, T, Q, method="trf")
popt_har, pcov_har = curve_fit(har_decline, T, Q, method="trf")

print("L2 Norm of exponential decline: ", L2_norm(exp_decline(T, popt_exp[0]), Q))
print("L2 Norm of hyperbolic decline decline: ", L2_norm(hyp_decline(T, popt_hyp[0], popt_hyp[1]), Q))
print("L2 Norm of harmonic decline decline: ", L2_norm(har_decline(T, popt_har[0]), Q))

fig, ax = plt.subplots(1, figsize=(10, 10))

ax.set_title("Decline Curve Analysis", fontsize=30)

label_size = 20
yed = [tick.label1.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
xed = [tick.label1.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]

ax.set_xlim(min(T), max(T))

ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
ax.set_xlabel("Time (years)", fontsize=25)
ax.set_ylabel("Oil Rate (1000 bbl/d)", fontsize=25)

pred_exp = exp_decline(T, popt_exp[0])
pred_hyp = hyp_decline(T, popt_hyp[0], popt_hyp[1])
pred_har = har_decline(T, popt_har[0])

min_val = min([min(curve) for curve in [pred_exp, pred_hyp, pred_har]])
max_val = max([max(curve) for curve in [pred_exp, pred_hyp, pred_har]])

ax.set_ylim(min_val, max_val)

ax.plot(T, pred_exp, color="red", linewidth=5, alpha=0.5, label="Exponential")
ax.plot(T, pred_hyp, color="green", linewidth=5, alpha=0.5, label="Hyperbolic")
ax.plot(T, pred_har, color="blue", linewidth=5, alpha=0.5, label="Harmonic")
ax.ticklabel_format(axis='both')
ax.legend(fontsize=25)

#prediction

fig, ax = plt.subplots(1, figsize=(10, 10))

T_max = 10.0
T_pred = np.linspace(min(T), T_max)

ax.set_title("Decline Curve Analysis Prediction", fontsize=30)

label_size = 20
yed = [tick.label.set_fontsize(label_size) for tick in ax.yaxis.get_major_ticks()]
xed = [tick.label.set_fontsize(label_size) for tick in ax.xaxis.get_major_ticks()]

ax.set_xlim(min(T), max(T_pred))

ax.scatter(T, Q, color="black", marker="x", s=250, linewidth=3)
ax.set_xlabel("Time (years)", fontsize=25)
ax.set_ylabel("Oil Rate (1000 bbl/d)", fontsize=25)

pred_exp = exp_decline(T_pred, popt_exp[0])
pred_hyp = hyp_decline(T_pred, popt_hyp[0], popt_hyp[1])
pred_har = har_decline(T_pred, popt_har[0])

min_val = min([min(curve) for curve in [pred_exp, pred_hyp, pred_har]])
max_val = max([max(curve) for curve in [pred_exp, pred_hyp, pred_har]])

ax.set_ylim(min_val, max_val)

ax.plot(T_pred, pred_exp, color="red", linewidth=5, alpha=0.5, label="Exponential")
ax.plot(T_pred, pred_hyp, color="green", linewidth=5, alpha=0.5, label="Hyperbolic")
# ax.plot(T_pred, pred_har, color="blue", linewidth=5, alpha=0.5, label="Harmonic")
ax.ticklabel_format(axis='both')
ax.legend(fontsize=25)
plt.show()