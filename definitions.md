### Baseline Formulas, improve or change as needed #### 

#### Network (adjacency_matrix.csv)

Default is a mixed graph (direct and undirected edges)

#### Non Normalized Risk Score with scaling factors: (alpha, beta, gamma, epsilon)

   R = (alpha * DegreeCentrality) + 
  (beta * BetweenessCentrality) + 
  (gamma * EigenvectorCentrality) +
  (epsilon * ClosenessCentrality)

#### Normalized Risk Score with scaling factors: (alpha, beta, gamma, epsilon)

   R_norm = (alpha * norm(DegreeCentrality)) + 
  (beta * norm(BetweenessCentrality)) + 
  (gamma * norm(EigenvectorCentrality)) +
  (epsilon * norm(ClosenessCentrality))  

#### Sigmoid transformation 

   Any transformation can be used if intuition is there is more nuance to be captured

   Sigmoid(R^*)

#### Final Risk

  p = the priority weight of the error type, which in this example
  will be either type-1 or type-2
  
  FR = R * (p * PriorityError + (1-p) * NonPriorityError)

#### Cost (Loss) (cost.csv) 

  This refers to the economic impact of the system, which is measured in units of value, where value can
  be any resource desired, (revenue, time saved, opportunity cost and so on). This would be the unit of value on success, and the destruction of value units on error, times the resource value per unit. 
  
  Where success = 1 (i.e. produces one unit of value), and an error is a negative value, 
  (e.g. -3 would mean an error negates 3 successful units of value). 
    
  C = FR * ErrorCost + (1−R) * SuccessValue 

#### Total Success Revenue

  Revenue unit is whatever dollar value assigned to a successful output. This could be assigned any desired value unit (time saved, opportunity cost etc.).
  
  Total Success Revenue (TS) = Final Success Rate * Success Value * Revenue Unit 

  Total Error Loss = Final Risk * Error Cost * Revenue Unit

  Net Revenue Impact = Total Success Revenue + Total Error Loss

#### Error Policy (error_policy.csv)

 The 'ignore' value set to 1 will ignore the error policy for that node and behave as a classic system output
 (with some lingering limited default uncertainty applied i.e. for considerting things like latency etc.),
 with all other risk metrics still applying to the node, i.e. this would be a node that is not producing transformer output, or any system part that is not producing a distribution of outcomes, e.g. event bridge etc.
   

#### Design Pattern Threats, Agentic or otherwise (design_threats.csv) 

 Security threats that a particular design pattern might introduce (e.g. jailbreaking etc.). These security threats can be applied per system component (node). Initial baseline threats were identifyed with Ken Haung's analysis, toy example threats were all just to set 0.5. 
 More threat types can be added and threat levels adjusted based on best approximation of reality. 

#### Resource Threats - Towers Of Knowlege, Churn, Depth (resource_threats.csv)

These are turnover and resource availability risk additions. This adds the potential personnel risk,
i.e. if a few engineers are responsible for large critical networks, and they are unhappy or
looking for other opportunities, or the company is restructuring, or this is a generally high churn rate 
associated with the companry, dept or team, this can help reflect this non-mechanistic risk.

            Multiplicative approach:
              Multiplicative: approach when interested in showing that increasing 
              resources dampen the overall risk relative to the baseline vulnerabilities. 
              Here, using churn_rate^resources_available gives the probability that ALL resources fail
              a lower value indicates better coverage and thus lower risk.
              resource_multiplier = churn_rate ** (resources_available/scaling_factor)
            
            Optional additive approach (this needs some adjusting):
              In this case if the analysis requires capturing two distinct sources of risk. 
              One inherent to the node (baseline) along with another coming from insufficient maintenance (churn).
              Survival probability is 1 - (churn_rate^resources_available/scaling_factor)

#### SLA Policy | Service Level Agreement (SLA.csv)
  * This would be a Service Level Agreement with any vendor or internal team, applied to any specific system component (node).
  
  * SLA uptime with actual uptime:\
  uptime_risk = (sla_uptime - actual_uptime) / sla_uptime\
  uptime_multiplier = 1 + (uptime_risk)

#### Notes
  Deterministic classic systems functionality can be compared against probabilistic nodes (transformers)
  where the stability of the classic node is compared against a less stable but possibily more value added
  transformer output.

  Will test implementing indicator functions for usesful/not useful output based on error policy thresholds.
  This gives us a simple Bernouli distribution where p = useful output/all output (per node, per system etc.). 
  From there p(1-p) is our variance and calculattions are straitforward, i.e. Fisher information etc.                    
