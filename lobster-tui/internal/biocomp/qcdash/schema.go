package qcdash

// QCDashboardData is the JSON payload for the qc_dashboard component.
type QCDashboardData struct {
	Metrics []QCMetric `json:"metrics"`
	Title   string     `json:"title,omitempty"`
}

// QCMetric represents a single quality metric.
type QCMetric struct {
	Name   string  `json:"name"`
	Value  float64 `json:"value"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Unit   string  `json:"unit,omitempty"`
	Status string  `json:"status,omitempty"` // "pass" | "warn" | "fail"
}
