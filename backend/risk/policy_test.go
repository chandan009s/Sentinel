package risk

import "testing"

func TestDetermineDecision(t *testing.T) {
	tests := []struct {
		name       string
		state      EventState
		riskLevel  string
		want       Decision
	}{
		{
			name:      "active incident blocks",
			state:     StateActive,
			riskLevel: "MEDIUM",
			want:      DecisionBlock,
		},
		{
			name:      "suspected high risk requires review",
			state:     StateSuspected,
			riskLevel: "HIGH",
			want:      DecisionReview,
		},
		{
			name:      "suspected medium risk allows",
			state:     StateSuspected,
			riskLevel: "MEDIUM",
			want:      DecisionAllow,
		},
		{
			name:      "recovery requires review",
			state:     StateRecovery,
			riskLevel: "LOW",
			want:      DecisionReview,
		},
		{
			name:      "resolved allows",
			state:     StateResolved,
			riskLevel: "HIGH",
			want:      DecisionAllow,
		},
		{
			name:      "normal allows",
			state:     StateNormal,
			riskLevel: "HIGH",
			want:      DecisionAllow,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetermineDecision(tt.state, tt.riskLevel)

			if got != tt.want {
				t.Fatalf("expected %s, got %s", tt.want, got)
			}
		})
	}
}