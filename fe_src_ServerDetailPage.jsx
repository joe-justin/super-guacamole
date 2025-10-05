import React from "react";
import { useParams } from "react-router-dom";
import ServerGraph from "../components/ServerGraph";

export default function ServerDetailPage() {
  const { server } = useParams();

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">{server}</h1>
      <ServerGraph server={server} />
    </div>
  );
}
