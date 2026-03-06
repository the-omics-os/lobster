package threshold

// ThresholdSliderData is the JSON payload for the threshold_slider component.
type ThresholdSliderData struct {
	Label   string  `json:"label"`
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`
	Step    float64 `json:"step"`
	Default float64 `json:"default"`
	Unit    string  `json:"unit,omitempty"`
	Count   int     `json:"count,omitempty"` // items passing at current value
	Total   int     `json:"total,omitempty"` // total items
}
