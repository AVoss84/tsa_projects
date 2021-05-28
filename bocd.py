from sklearn.base import BaseEstimator, OutlierMixin
from scipy.stats import norm
from scipy.special import logsumexp
from tqdm import tqdm


class BayesOCPD(BaseEstimator, OutlierMixin):
    """
    Bayesian online change point detection, see Adams & MacKay 2007.
    """
    def __init__(self, model, hazard, mini_run_length : int = 10, verbose : bool = True):
        self.verbose = verbose
        self.model = model
        self.hazard = hazard
        self.mini_run_length = mini_run_length
        if self.verbose : print("*** Bayesian online change point detection ***")

    def fit(self, X, y=None):    
        """
        Return run length posterior using Algorithm 1
        """
        data = deepcopy(X)
        # 1. Initialize
        T           = len(data)
        log_R       = -np.inf * np.ones((T+1, T+1))    # log posterior values of run length at time t
        log_R[0, 0] = 0              # log 0 == 1
        pmean       = np.empty(T)    # Model's predictive mean.
        pvar        = np.empty(T)    # Model's predictive variance. 
        log_message = np.array([0])  # log 0 == 1
        log_H       = np.log(self.hazard)
        log_1mH     = np.log(1 - self.hazard)
        cps_MAP     = np.empty(T) 

        for t in tqdm(range(1, T+1)):       # forward filtering 
            # 2. Observe new datum.
            x = data[t-1]

            # Make model predictions. ('step 9'). Calculate first two moments
            pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * self.model.mean_params[:t])   # up to t-1
            pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * self.model.var_params[:t])
            
            # 3. Evaluate predictive posterior probabilities.
            log_pis = model.log_pred_prob(t, x)

            # 4. Calculate growth probabilities ('No break')
            log_growth_probs = log_pis + log_message + log_1mH

            # 5. Calculate changepoint/Break probabilities.
            log_cp_prob = logsumexp(log_pis + log_message + log_H)

            # 6. Calculate evidence
            new_log_joint = np.append(log_cp_prob, log_growth_probs)    # append: pr(r_t == 0), pr(r_t > 0)'s

            # 7. Determine run length posterior mass distribution.
            log_R[t, :t+1]  = new_log_joint
            log_R[t, :t+1] -= logsumexp(new_log_joint)            # normalize entries by dividing by column sum

            # 8. Update sufficient statistics.
            # here: mean_params & prec_params
            model.update_params(t, x)

            # Pass message.
            log_message = new_log_joint

            # Select break points, via MAP state estimate        
            cps_MAP[t-1] = np.argmax(log_R[t-1,:t])
        
        self.pmean = pmean
        self.pvar = pvar
        self.R = np.exp(log_R)
        self.log_R = log_R
        self.cps_MAP = cps_MAP
        return self
    
    # Future: change this to be able run for online oob samples
    def predict(self, X):
        lab = np.ones(len(X))
        lab[self.score(X)] = -1
        return lab

    def score(self, X):
        #return np.amax(self.R, 1)    
        cps_est = np.where((self.cps_MAP[1:] - self.cps_MAP[:-1]) < -self.mini_run_length)[0]    # MAP estimates of change points
        return cps_est
