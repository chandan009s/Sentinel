package risk

type Decision string

const (
	DecisionAllow  Decision = "ALLOW"
	DecisionReview Decision = "REVIEW"
	DecisionBlock  Decision = "BLOCK"
)

func DetermineDecision(eventState EventState, riskLevel string) Decision {
	switch eventState {
	case StateActive:
		return DecisionBlock

	case StateSuspected:
		if riskLevel == "HIGH" {
			return DecisionReview
		}
		return DecisionAllow

	case StateRecovery:
		return DecisionReview

	case StateResolved:
		return DecisionAllow

	default:
		return DecisionAllow
	}
}
