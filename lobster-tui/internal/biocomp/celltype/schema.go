package celltype

// CellTypeSelectorData is the JSON payload for the cell_type_selector component.
type CellTypeSelectorData struct {
	Clusters []ClusterInfo `json:"clusters"`
}

// ClusterInfo represents one cluster with markers and an optional pre-filled label.
type ClusterInfo struct {
	ID      int      `json:"id"`
	Size    int      `json:"size"`
	Markers []string `json:"markers"`
	Label   string   `json:"label,omitempty"` // pre-filled suggestion from LLM
}
