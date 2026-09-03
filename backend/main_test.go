package main

import "testing"

func TestDetermineRiskLevel(t *testing.T) {
	threshold := 0.6577208933126732

	tests := []struct {
		name      string
		score     float64
		threshold float64
		expected  string
	}{
		{
			name:      "low risk",
			score:     0.382,
			threshold: threshold,
			expected:  "LOW",
		},
		{
			name:      "medium risk",
			score:     0.68,
			threshold: threshold,
			expected:  "MEDIUM",
		},
		{
			name:      "high risk",
			score:     0.814,
			threshold: threshold,
			expected:  "HIGH",
		},
		{
			name:      "below threshold",
			score:     0.65,
			threshold: threshold,
			expected:  "LOW",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := determineRiskLevel(tt.score, tt.threshold)

			if got != tt.expected {
				t.Errorf(
					"determineRiskLevel(%v, %v) = %q, want %q",
					tt.score,
					tt.threshold,
					got,
					tt.expected,
				)
			}
		})
	}
}
