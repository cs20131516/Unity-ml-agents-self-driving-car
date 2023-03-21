using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class CarAgent : Agent
{
    private Transform tr;
    private Rigidbody rb;

    public float moveSpeed;
    public float Turn;
    Vector3 startPosition;
    Vector3 startRotation;

    private int numStep = 0;

    public override void Initialize()
    {
        tr = GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        startPosition = tr.position;
        startRotation = tr.eulerAngles;
    }

    public override void OnEpisodeBegin()
    {
        // 물리엔진 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        //tr.localPosition = startPosition;
        //tr.localEulerAngles=startRotation;
        // 위치 초기화
        tr.position = startPosition;
        tr.eulerAngles = startRotation;
    }

    public override void CollectObservations(VectorSensor sensor)
    {

    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var action = actions.DiscreteActions[0];

        float frontDist = 0;
        float leftDist = 0;
        float rightDist = 0;
        float dist = 0;
        // 정면 : rayIndex = 0
        // 오른쪽 : rayIndex = 1
        // 왼쪽 : rayIndex = 2
        // Ray Length = 4
        // 좌우 ray length = 4 * 1.5 = 6
        var raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();
        var input = raySensorComponent.GetRayPerceptionInput();
        var outputs = RayPerceptionSensor.Perceive(input);
        for (var rayIndex = 0; rayIndex < outputs.RayOutputs.Length; rayIndex++)
        {
            var extents = input.RayExtents(rayIndex);
            Vector3 startPositionWorld = extents.StartPositionWorld;
            Vector3 endPositionWorld = extents.EndPositionWorld;

            //Vector3 startPositionWorld_1 = extents.StartPositionWorld + new Vector3(1.0f, 0.0f, 1.0f);
            //Vector3 endPositionWorld_1 = extents.EndPositionWorld + new Vector3(1.0f, 0.0f, 1.0f);


            var rayOutput = outputs.RayOutputs[rayIndex];
            if (rayOutput.HasHit)
            {
                Vector3 hitPosition = Vector3.Lerp(startPositionWorld, endPositionWorld, rayOutput.HitFraction);
                //Debug.DrawLine(startPositionWorld, hitPosition, Color.red);
                //Debug.Log(rayIndex + " " + Vector3.Distance(hitPosition, startPositionWorld));

                if (rayIndex == 0)   frontDist = Vector3.Distance(hitPosition, startPositionWorld);
                else if(rayIndex==1) rightDist = Vector3.Distance(hitPosition, startPositionWorld);
                else if(rayIndex==2) leftDist = Vector3.Distance(hitPosition, startPositionWorld);
                //Vector3 hitPosition_1 = Vector3.Lerp(startPositionWorld_1, endPositionWorld_1, rayOutput.HitFraction);
                //Debug.DrawLine(startPositionWorld_1, hitPosition_1, Color.red);
                //Debug.Log(rayIndex + " " + Vector3.Distance(hitPosition_1, startPositionWorld_1));
                
            }
            else
            {
                if (rayIndex == 0) frontDist = Vector3.Distance(endPositionWorld, startPositionWorld);
                else if (rayIndex == 1) rightDist = Vector3.Distance(endPositionWorld, startPositionWorld);
                else if (rayIndex == 2) leftDist = Vector3.Distance(endPositionWorld, startPositionWorld);
                //Debug.Log(rayIndex + " " + Vector3.Distance(endPositionWorld, startPositionWorld));
            }
        }
        dist = leftDist - rightDist;
        //Debug.Log("Ray distance : " + dist);
        // 흰선 밟았을 때
        if (dist > 4.8084f)                         SetReward(-0.04f);
        else if (dist > 3.2392f && dist <= 4.8084f) SetReward(-0.02f);
        else if (dist > 2.7203f && dist <= 3.2392f) SetReward(-0.01f);
        // 정상 주행
        else if (dist > 2.2075f && dist <= 2.7203f) SetReward(0.005f);
        // 중앙선 밟았을 때
        else if (dist > 2.0387f && dist <= 2.2075f) SetReward(-0.03f);
        else if (dist > 1.2944f && dist <= 2.0387f) SetReward(-0.05f);
        else if (dist <= 1.2944f)                   SetReward(-0.1f);
        

        //var rayComponent = GetComponent<RayPerceptionSensorComponent3D>();

        //var input = rayComponent.GetRayPerceptionInput();
        //var output = RayPerceptionSensor.Perceive(input);

        //float frontDist = 0;
        //float leftDist = 0;
        //float rightDist = 0;

        //// 정면 : rayIndex = 0
        //// 오른쪽 : rayIndex = 1
        //// 왼쪽 : rayIndex = 2
        //for (var rayIndex = 0; rayIndex < output.RayOutputs.Length; rayIndex++)
        //{
        //    var extents = input.RayExtents(rayIndex);
        //    Vector3 startPositionRay = extents.StartPositionWorld;
        //    Vector3 endPositionRay = extents.EndPositionWorld;
        //    var rayOutput = output.RayOutputs[rayIndex];
        //    Vector3 hitPosition = Vector3.Lerp(startPositionRay, endPositionRay, rayOutput.HitFraction);

        //    float hit = Vector3.Distance(hitPosition, startPositionRay);
        //    float nonhit = Vector3.Distance(endPositionRay, startPositionRay);

        //    if (rayOutput.HasHit)
        //    {
        //        Debug.DrawLine(startPositionRay, hitPosition, Color.green);
        //        print("hit : " + rayIndex);
        //        Debug.Log(Vector3.Distance(hitPosition, startPositionRay));

        //        if (rayIndex == 0) frontDist = hit;
        //        else if (rayIndex == 1) rightDist = hit;
        //        else if (rayIndex == 2) leftDist = hit;
        //    }
        //    else
        //    {
        //        Debug.DrawLine(startPositionRay, endPositionRay, Color.black);
        //        print("nonhit : " + rayIndex);
        //        Debug.Log(Vector3.Distance(endPositionRay, startPositionRay));

        //        if (rayIndex == 0) frontDist = nonhit;
        //        else if (rayIndex == 1) rightDist = nonhit;
        //        else if (rayIndex == 2) leftDist = nonhit;
        //    }
        //}

        //if (frontDist > 25 && rightDist > 13 && leftDist > 13) AddReward(0.1f);

        switch (action)
        {
            case 0: Turn = 0f; moveSpeed = 5.0f; break;
            case 1: Turn = 0f; moveSpeed = 7.5f; break;
            case 2: Turn = 0f; moveSpeed = 10.0f; break;
            case 3: Turn = -0.7f; moveSpeed = 5.0f; break;
            case 4: Turn = -0.7f; moveSpeed = 7.5f; break;
            case 5: Turn = -0.7f; moveSpeed = 10.0f; break;
            case 6: Turn = 0.7f; moveSpeed = 5.0f; break;
            case 7: Turn = 0.7f; moveSpeed = 7.5f; break;
            case 8: Turn = 0.7f; moveSpeed = 10.0f; break;
            case 9: Turn = -1.4f; moveSpeed = 5.0f; break;
            case 10: Turn = -1.4f; moveSpeed = 7.5f; break;
            case 11: Turn = -1.4f; moveSpeed = 10.0f; break;
            case 12: Turn = 1.4f; moveSpeed = 5.0f; break;
            case 13: Turn = 1.4f; moveSpeed = 7.5f; break;
            case 14: Turn = 1.4f; moveSpeed = 10.0f; break;
        }

        transform.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward);
        transform.Rotate(new Vector3(0f, Turn, 0f));

        numStep += 1;
        Debug.Log("step : " + numStep);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actionOut = actionsOut.DiscreteActions[0];
        // 왼쪽
        if (Input.GetKey(KeyCode.LeftArrow)) actionOut = 9;

        // 직진
        if (Input.GetKey(KeyCode.UpArrow)) actionOut = 0;

        // 오른쪽
        if (Input.GetKey(KeyCode.RightArrow)) actionOut = 12;
    }

    private void OnCollisionEnter(Collision collision)
    {
        numStep = 0;

        if (collision.collider.CompareTag("wall"))
        {
            SetReward(-1.0f);
            EndEpisode();
        }
        if(collision.collider.CompareTag("goal"))
        {
            SetReward(2.0f);
            EndEpisode();
        }
    }
}
